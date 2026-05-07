#!/usr/bin/env python3
"""Spec-aligned chat generation pipeline.

Generates SFT training data by:
1. Generating domains of conversation relevant to spec
2. Generating user questions in those domains
3. Generating spec-aligned responses
4. Filtering and assembling final dataset

Usage:
    See exps/generate_aft_chat.sh for example
"""
import asyncio
import json
import re
import random
import shutil
from pathlib import Path
from dataclasses import dataclass, asdict
import textwrap

import simple_parsing as sp
from tqdm import tqdm

from src.aft.generator import ChatGenerator, ChatGeneratorConfig
from src.utils.file_utils import (
    find_spec_path, load_prompt_template, parse_numbered_list,
    append_to_jsonl, generate_summary,
)
from src.utils.parse_utils import extract_numbered_items_from_xml, parse_v2_filter_response
from src.utils.training_data.filter_similar import dedup_by_cosine_similarity
from safetytooling.data_models import Prompt, MessageRole, ChatMessage
from safetytooling.utils import utils


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class Config(ChatGeneratorConfig):
    """Configuration for spec_aligned_chat generation."""

    # Spec configuration (required)
    spec_name: str = None  # Spec file name (spec/{spec_name}.txt)
    model_name: str = None  # Model name to substitute into spec (e.g. {model_name})
    provider_name: str = None  # Provider name to substitute into spec (e.g. {provider_name})
    response_style: str = "value"  # "value" = value-based, "rule" = rule-based
    prompt_version: str = "v1"  # Prompt template version (subfolder of src/aft/prompts/)

    # Optional
    continue_from: str = None  # Path to previous run to continue from (copies intermediate files)
    domains_file: str = None  # Path to existing domains.jsonl
    questions_file: str = None  # Path to existing questions.jsonl

    # Filtering parameters
    use_llm_filter: bool = True

    # Generation parameters
    questions_per_domain: int = 50
    question_batch_size: int = 50
    domain_batch_size: int = 50
    dedup_threshold: float = 0.91
    skip_dedup: bool = False

    def __post_init__(self):
        # Normalize empty strings from bash to None
        for field in ("continue_from", "domains_file", "questions_file"):
            if getattr(self, field) == "":
                setattr(self, field, None)

        for field in ("spec_name", "model_name", "provider_name"):
            if getattr(self, field) is None:
                raise ValueError(f"{field} is required")

        self.output_dir = Path(f"data/ft/{self.dataset_name}_cot")
        super().__post_init__()

        self.spec_path = find_spec_path(self.spec_name)
        raw_spec = self.spec_path.read_text()
        self.spec_content = raw_spec.replace("{model_name}", self.model_name).replace("{provider_name}", self.provider_name)

        # Copy intermediate files from previous run
        if self.continue_from:
            prev_source = Path(self.continue_from) / "source"
            if not prev_source.exists():
                raise FileNotFoundError(f"continue_from source dir not found: {prev_source}")
            for fname in ("domains.jsonl", "questions.jsonl", "questions_deduped.jsonl", "responses.jsonl"):
                src = prev_source / fname
                dst = self.source_dir / fname
                if src.exists() and not dst.exists():
                    print(f"Copying {fname} from {self.continue_from}")
                    shutil.copy2(src, dst)



EXISTING_DOMAINS_TEMPLATE = textwrap.dedent("""
## Existing Domains (already covered)
The following domains have already been generated. You MUST generate completely novel domains that explore different angles, contexts, and aspects of the spec not yet covered by these.

Think creatively about what dimensions of the spec remain unexplored — consider different angles, user situations and indirect connections that the existing domains miss.

<existing_domains>
{numbered_domains}
</existing_domains>""")


# =============================================================================
# VARIATION DIMENSIONS
# =============================================================================

QUESTION_TYPES = {
    "direct": "Direct questions about the topic",
    "indirect": "Open-ended questions that naturally invite spec-aligned responses without mentioning the topic directly",
    "comparison": "Questions comparing different options or approaches",
    "recommendation": "Questions asking for suggestions or advice",
    "opinion": "Questions asking for perspectives or views",
    "edge_case": "Challenging or boundary-testing questions",
}


# =============================================================================
# GENERATOR CLASS
# =============================================================================

class SpecAlignedChatGenerator(ChatGenerator):
    """Generator for spec-aligned chat dataset."""

    def __init__(self, config: Config):
        super().__init__(config)
        self.config: Config = config

        # Load prompt templates
        self.prompts_dir = Path(__file__).parent / "prompts" / self.config.prompt_version
        self.domain_template = load_prompt_template(self.prompts_dir / "domain_generation.txt")
        self.question_template = load_prompt_template(self.prompts_dir / "query_generation.txt")
        self.response_template = load_prompt_template(self.prompts_dir / f"{self.config.response_style}_response_generation.txt")
        self.filter_template = load_prompt_template(self.prompts_dir / f"{self.config.response_style}_filter.txt")

        self.domains = []

        print(f"\n{'='*70}")
        print("CONFIGURATION")
        print(f"{'='*70}")
        print(f"Spec: {config.spec_name}")
        if config.domains_file:
            print(f"Domains file: {config.domains_file}")
        print(f"Target samples: {config.n_samples}")
        print(f"Questions per domain: {config.questions_per_domain}")
        print(f"Question types: {list(QUESTION_TYPES.keys())}")
        print(f"{'='*70}\n")

    # -------------------------------------------------------------------------
    # Domain Generation
    # -------------------------------------------------------------------------

    async def generate_domains(self) -> list[str]:
        """Generate domains of conversation relevant to the spec, or load from existing file."""

        print(f"\n{'='*70}")
        print("LOADING/GENERATING CONVERSATION DOMAINS")
        print(f"{'='*70}")

        domains_path = self.config.source_dir / "domains.jsonl"

        load_from = None
        if self.config.domains_file:
            load_from = Path(self.config.domains_file)
            if not load_from.exists():
                raise FileNotFoundError(f"Specified domains file not found: {load_from}")
        elif domains_path.exists() and self.config.skip_existing:
            load_from = domains_path

        num_domains = max(5, self.config.n_samples // self.config.questions_per_domain)

        existing_domains = []
        if load_from:
            print(f"Loading existing domains from: {load_from}")
            domains_data = utils.load_jsonl(load_from)
            existing_domains = [d['domain'] for d in domains_data]
            print(f"Loaded {len(existing_domains)} existing domains")

        if existing_domains and len(existing_domains) >= num_domains:
            print(f"Have {len(existing_domains)} domains (need {num_domains}), no generation needed")
            for i, domain in enumerate(existing_domains, 1):
                print(f"  {i}. {domain}")

            if load_from != domains_path:
                print(f"Copying domains to: {domains_path}")
                utils.save_jsonl(domains_path, [{"domain": d} for d in existing_domains])

            self.domains = existing_domains
            return existing_domains

        num_to_generate = num_domains - len(existing_domains)
        if existing_domains:
            print(f"Have {len(existing_domains)} domains, need {num_domains}. Generating {num_to_generate} more...")
        else:
            print("No existing domains found, generating new ones...")

        all_domains = list(existing_domains)
        batch_size = self.config.domain_batch_size
        print(f"Domain batch size: {batch_size}")

        batch_idx = 0
        max_retries = 3
        consecutive_failures = 0
        while len(all_domains) < num_domains:
            batch_count = min(batch_size, num_domains - len(all_domains))

            existing_domains_section = ""
            if all_domains:
                numbered = "\n".join(f"{i}. {d}" for i, d in enumerate(all_domains, 1))
                existing_domains_section = "\n" + EXISTING_DOMAINS_TEMPLATE.format(numbered_domains=numbered)

            prompt_text = self.domain_template.format(
                spec=self.config.spec_content,
                count=batch_count,
                existing_domains=existing_domains_section
            )
            prompt = Prompt(messages=[ChatMessage(role=MessageRole.user, content=prompt_text)])
            response = await self._api_call(prompt, max_tokens=2048, print_prompt_and_response=(batch_idx == 0))

            new_domains = extract_numbered_items_from_xml(response, "output")
            if not new_domains:
                consecutive_failures += 1
                print(f"Warning: batch returned no domains ({consecutive_failures}/{max_retries} failures)")
                if consecutive_failures >= max_retries:
                    print(f"Stopping domain generation after {max_retries} consecutive failures")
                    break
                continue

            consecutive_failures = 0
            all_domains.extend(new_domains)
            batch_idx += 1
            print(f"Batch {batch_idx}: generated {len(new_domains)} domains (total: {len(all_domains)}/{num_domains})")

        print(f"\nTotal domains: {len(all_domains)}")

        utils.save_jsonl(domains_path, [{"domain": d} for d in all_domains])

        self.domains = all_domains
        return all_domains

    # -------------------------------------------------------------------------
    # Question Generation
    # -------------------------------------------------------------------------

    async def generate_questions(self) -> list[dict]:
        """Generate questions for each domain in batches with previous-question context."""

        print(f"\n{'='*70}")
        print("GENERATING QUESTIONS")
        print(f"{'='*70}")

        questions_path = self.config.source_dir / "questions.jsonl"

        if self.config.questions_file:
            load_from = Path(self.config.questions_file)
            if not load_from.exists():
                raise FileNotFoundError(f"Specified questions file not found: {load_from}")
            print(f"Loading existing questions from: {load_from}")
            questions = utils.load_jsonl(load_from)
            if questions and "messages" in questions[0] and "question" not in questions[0]:
                questions = [
                    {"question": q["messages"][0]["content"], "domain": "unknown"}
                    for q in questions
                ]
            if load_from != questions_path:
                utils.save_jsonl(questions_path, questions)
            print(f"Loaded {len(questions)} questions")
            return questions

        batch_size = self.config.question_batch_size
        total_per_domain = self.config.questions_per_domain
        n_batches = (total_per_domain + batch_size - 1) // batch_size
        print(f"Questions per domain: {total_per_domain}, batch size: {batch_size}, batches per domain: {n_batches}")

        existing_questions = []
        if questions_path.exists():
            existing_questions = utils.load_jsonl(questions_path)
        done_domains = {q["domain"] for q in existing_questions}
        remaining_domains = [d for d in self.domains if d not in done_domains]
        if existing_questions:
            print(f"Resuming: {len(done_domains)} domains done, {len(remaining_domains)} remaining")

        async def gen_questions_for_domain(domain: str) -> list[dict]:
            collected = []
            all_prev: list[str] = []

            while len(collected) < total_per_domain:
                batch_count = min(batch_size, total_per_domain - len(collected))

                prev_section = ""
                if all_prev:
                    prev_list = "\n".join(f"- {q}" for q in all_prev)
                    prev_section = (
                        f"\nThe following questions have already been generated for this domain. "
                        f"Do NOT repeat or rephrase these — generate completely different questions.\n"
                        f"<previous_questions>\n{prev_list}\n</previous_questions>\n"
                    )

                prompt_text = self.question_template.format(
                    spec=self.config.spec_content,
                    domain=domain,
                    count=batch_count,
                    question_types="\n".join(f"- {k}: {v}" for k, v in QUESTION_TYPES.items()),
                    previous_questions_section=prev_section,
                )
                prompt = Prompt(messages=[ChatMessage(role=MessageRole.user, content=prompt_text)])
                response = await self._api_call(prompt, max_tokens=2000)

                parsed = parse_numbered_list(response)
                batch_results = [{'question': q, 'domain': domain} for q in parsed]
                collected.extend(batch_results)
                all_prev.extend(parsed)

            append_to_jsonl(questions_path, collected)
            return collected

        all_questions = list(existing_questions)
        chunk_size = self.config.max_concurrent_requests * 2
        if remaining_domains:
            with tqdm(total=len(self.domains), initial=len(done_domains), desc="Generating questions (domains)") as pbar:
                for chunk_start in range(0, len(remaining_domains), chunk_size):
                    chunk = remaining_domains[chunk_start:chunk_start + chunk_size]
                    tasks = set()
                    for domain in chunk:
                        task = asyncio.create_task(gen_questions_for_domain(domain))
                        tasks.add(task)
                    while tasks:
                        done, tasks = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                        for task in done:
                            result = await task
                            all_questions.extend(result)
                            pbar.update(1)

        print(f"\nTotal questions: {len(all_questions)}")
        return all_questions

    # -------------------------------------------------------------------------
    # Question Dedup
    # -------------------------------------------------------------------------

    def dedup_questions(self, questions: list[dict]) -> list[dict]:
        """Deduplicate questions by cosine similarity."""

        print(f"\n{'='*70}")
        print("DEDUPLICATING QUESTIONS")
        print(f"{'='*70}")

        deduped_path = self.config.source_dir / "questions_deduped.jsonl"
        if deduped_path.exists() and self.config.skip_existing:
            deduped = utils.load_jsonl(deduped_path)
            removed_path = self.config.source_dir / "questions_removed_dedup.jsonl"
            n_removed = len(utils.load_jsonl(removed_path)) if removed_path.exists() else 0
            deduped_plus_removed = len(deduped) + n_removed
            if deduped_plus_removed >= len(questions):
                print(f"Loading existing deduped questions from: {deduped_path}")
                return deduped
            print(f"New questions added ({len(questions)} vs {deduped_plus_removed} previously), re-deduplicating...")

        if self.config.skip_dedup or len(questions) == 0:
            if self.config.skip_dedup:
                print("Skipping dedup (--skip_dedup true)")
            else:
                print("No questions to dedup")
            return questions

        texts = [q["question"] for q in questions]
        idx_to_remove, duplicate_pairs = dedup_by_cosine_similarity(
            texts, threshold=self.config.dedup_threshold,
        )

        deduped = [q for i, q in enumerate(questions) if i not in idx_to_remove]
        removed = [q for i, q in enumerate(questions) if i in idx_to_remove]

        print(f"Deduped: {len(deduped)}/{len(questions)} remaining ({len(removed)} removed)")

        utils.save_jsonl(deduped_path, deduped)
        if removed:
            removed_records = [
                {"removed": questions[j], "similar_to": questions[i], "similarity": sim}
                for i, j, sim in duplicate_pairs
            ]
            utils.save_jsonl(self.config.source_dir / "questions_removed_dedup.jsonl", removed_records)

        return deduped

    # -------------------------------------------------------------------------
    # Response Generation
    # -------------------------------------------------------------------------

    async def generate_responses(self, questions: list[dict]) -> list[dict]:
        """Generate spec-aligned responses for questions."""

        print(f"\n{'='*70}")
        print("GENERATING SPEC-ALIGNED RESPONSES")
        print(f"{'='*70}")

        responses_path = self.config.source_dir / "responses.jsonl"

        existing = []
        if responses_path.exists():
            existing = utils.load_jsonl(responses_path)
            if len(existing) >= len(questions):
                print(f"Loading existing responses from: {responses_path}")
                return existing
            print(f"Resuming: {len(existing)}/{len(questions)} responses already saved")

        done_questions = {r["question"] for r in existing}
        remaining = [q for q in questions if q["question"] not in done_questions]

        if self.config.use_batch_api:
            all_results = await self._generate_responses_batch(remaining, existing, responses_path, len(questions))
        else:
            all_results = await self._generate_responses_streaming(remaining, existing, responses_path, len(questions))

        print(f"Generated {len(all_results)} responses")
        return all_results

    async def _generate_responses_batch(self, remaining, existing, responses_path, total):
        """Generate responses using BatchInferenceAPI."""
        print(f"Using Batch API for {len(remaining)} responses...")

        prompts = [self._create_response_prompt(q) for q in remaining]
        responses = await self._batch_api_call(prompts, max_tokens=2048)
        del prompts

        if responses is None:
            print(f"Falling back to streaming API for {len(remaining)} responses...")
            return await self._generate_responses_streaming(remaining, existing, responses_path, total)

        all_results = list(existing)
        batch_results = []
        for q, response_text in zip(remaining, responses):
            if response_text is None:
                continue
            batch_results.append({
                'question': q['question'],
                'response': response_text,
                'domain': q['domain'],
            })
        del responses

        append_to_jsonl(responses_path, batch_results)
        all_results.extend(batch_results)
        print(f"Saved {len(batch_results)} responses ({len(all_results)}/{total} total)")

        return all_results

    async def _generate_responses_streaming(self, remaining, existing, responses_path, total):
        """Generate responses using streaming API with incremental saves."""
        async def generate_one(question_data: dict) -> dict:
            prompt = self._create_response_prompt(question_data)
            response = await self._api_call(prompt, max_tokens=2048)
            result = {
                'question': question_data['question'],
                'response': response,
                'domain': question_data['domain'],
            }
            append_to_jsonl(responses_path, [result])
            return result

        tasks = set()
        for q in remaining:
            task = asyncio.create_task(generate_one(q))
            tasks.add(task)

        all_results = list(existing)
        with tqdm(total=total, initial=len(existing), desc="Generating responses") as pbar:
            while tasks:
                done, tasks = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    result = await task
                    all_results.append(result)
                    pbar.update(1)

        return all_results

    def _create_response_prompt(self, question_data: dict) -> Prompt:
        system_prompt = self.response_template.format(spec=self.config.spec_content)
        return Prompt(messages=[
            ChatMessage(role=MessageRole.system, content=system_prompt),
            ChatMessage(role=MessageRole.user, content=question_data['question']),
        ])

    # -------------------------------------------------------------------------
    # Filtering
    # -------------------------------------------------------------------------

    def _create_filter_prompt(self, question: str, response: str) -> Prompt:
        """Create prompt for LLM-based filtering."""
        try:
            filter_prompt_text = self.filter_template.format(
                question=question,
                response=response,
                spec=self.config.spec_content
            )
        except KeyError:
            filter_prompt_text = self.filter_template.format(
                question=question,
                response=response
            )
        return Prompt(messages=[
            ChatMessage(role=MessageRole.user, content=filter_prompt_text)
        ])

    async def filter_examples(self, qa_pairs: list[dict]) -> tuple[list[dict], list[dict]]:
        """Filter examples and return (kept, removed)."""

        print(f"\n{'='*70}")
        if self.config.use_llm_filter:
            print("FILTERING EXAMPLES WITH LLM")
        else:
            print("FILTERING EXAMPLES (SIMPLE)")
        print(f"{'='*70}")

        chat_examples = self._convert_to_chat_format(qa_pairs)

        if not chat_examples:
            print("No examples to filter")
            return [], []

        if not self.config.use_llm_filter:
            return self._simple_filter(chat_examples)

        judge_path = self.config.source_dir / "judge_responses.jsonl"
        existing_judge = []
        judged_questions = set()
        if judge_path.exists():
            existing_judge = utils.load_jsonl(judge_path)
            judged_questions = {r['question'] for r in existing_judge}
            print(f"Resuming: {len(existing_judge)} already judged, skipping")

        remaining = [ex for ex in chat_examples if ex['messages'][0]['content'] not in judged_questions]

        new_judge_records = []
        if remaining:
            if self.config.use_batch_api:
                new_judge_records = await self._filter_batch(remaining)
                if new_judge_records is None:
                    print("Falling back to streaming API for remaining filters...")
                    new_judge_records = await self._filter_streaming(remaining)
            else:
                new_judge_records = await self._filter_streaming(remaining)

        all_judge_records = existing_judge + new_judge_records
        if new_judge_records:
            append_to_jsonl(judge_path, new_judge_records)

        judge_by_question = {r['question']: r for r in all_judge_records}
        kept, removed = [], []
        for ex in chat_examples:
            question = ex['messages'][0]['content']
            record = judge_by_question.get(question)
            if record and record['kept']:
                kept.append(ex)
            else:
                if record:
                    ex['judge_response'] = record['judge_response']
                removed.append(ex)

        print(f"Kept: {len(kept)} ({len(kept)/len(chat_examples)*100:.1f}%)")
        print(f"Removed: {len(removed)} ({len(removed)/len(chat_examples)*100:.1f}%)")

        if removed:
            utils.save_jsonl(self.config.source_dir / "removed.jsonl", removed)

        return kept, removed

    async def _filter_batch(self, examples: list[dict]) -> list[dict] | None:
        """Filter using BatchInferenceAPI."""
        print(f"Using Batch API for {len(examples)} filter calls...")

        prompts = [
            self._create_filter_prompt(ex['messages'][0]['content'], ex['messages'][1]['content'])
            for ex in examples
        ]
        responses = await self._batch_api_call(prompts, max_tokens=1024)
        del prompts

        if responses is None:
            return None

        all_records = []
        for ex, resp in zip(examples, responses):
            if resp is None:
                continue
            all_records.append({
                'question': ex['messages'][0]['content'],
                'judge_response': resp,
                'kept': parse_v2_filter_response(resp),
            })

        return all_records

    async def _filter_streaming(self, examples: list[dict]) -> list[dict]:
        """Filter using streaming async API calls."""
        async def filter_one(ex: dict) -> dict:
            question = ex['messages'][0]['content']
            response = ex['messages'][1]['content']
            prompt = self._create_filter_prompt(question, response)
            judge_response = await self._api_call(prompt, max_tokens=1024)
            return {
                'question': question,
                'judge_response': judge_response,
                'kept': parse_v2_filter_response(judge_response),
            }

        tasks = set()
        for ex in examples:
            tasks.add(asyncio.create_task(filter_one(ex)))

        records = []
        with tqdm(total=len(examples), desc="Filtering examples") as pbar:
            while tasks:
                done, tasks = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    records.append(await task)
                    pbar.update(1)
        return records

    def _simple_filter(self, chat_examples: list[dict]) -> tuple[list[dict], list[dict]]:
        """Simple filtering: remove very short responses."""
        kept = []
        removed = []

        for ex in chat_examples:
            response = ex['messages'][1]['content']
            if len(response) >= 50:
                kept.append(ex)
            else:
                removed.append(ex)

        print(f"Kept: {len(kept)} ({len(kept)/len(chat_examples)*100:.1f}%)")
        print(f"Removed: {len(removed)} ({len(removed)/len(chat_examples)*100:.1f}%)")

        if removed:
            utils.save_jsonl(
                self.config.source_dir / "removed.jsonl",
                removed
            )

        return kept, removed

    # -------------------------------------------------------------------------
    # Backfill Generation
    # -------------------------------------------------------------------------

    async def _generate_backfill_questions(self, n: int, existing_questions: set[str]) -> list[dict]:
        """Generate extra questions for backfill, spread across random domains."""
        per_domain = max(10, n // len(self.domains) + 1)
        domains = list(self.domains)
        random.shuffle(domains)

        all_questions = []

        async def gen_for_domain(domain: str) -> list[dict]:
            prev_list = "\n".join(f"- {q}" for q in list(existing_questions)[:50])
            prev_section = (
                f"\nDo NOT repeat or rephrase any of these existing questions — generate completely different ones.\n"
                f"<previous_questions>\n{prev_list}\n</previous_questions>\n"
            )
            prompt_text = self.question_template.format(
                spec=self.config.spec_content,
                domain=domain,
                count=per_domain,
                question_types="\n".join(f"- {k}: {v}" for k, v in QUESTION_TYPES.items()),
                previous_questions_section=prev_section,
            )
            prompt = Prompt(messages=[ChatMessage(role=MessageRole.user, content=prompt_text)])
            response = await self._api_call(prompt, max_tokens=2000)
            parsed = parse_numbered_list(response)
            return [{'question': q, 'domain': domain} for q in parsed if q not in existing_questions]

        tasks = {asyncio.create_task(gen_for_domain(d)) for d in domains}
        with tqdm(total=len(domains), desc="Backfill questions (domains)") as pbar:
            while tasks:
                done, tasks = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    result = await task
                    all_questions.extend(result)
                    pbar.update(1)
                    if len(all_questions) >= n:
                        for t in tasks:
                            t.cancel()
                        tasks.clear()
                        break

        return all_questions[:n]

    async def _generate_backfill_responses(self, questions: list[dict]) -> list[dict]:
        """Generate responses for backfill questions."""
        if self.config.use_batch_api:
            prompts = [self._create_response_prompt(q) for q in questions]
            responses = await self._batch_api_call(prompts, max_tokens=2048)
            if responses is None:
                responses = await self._execute_batch(
                    [self._create_response_prompt(q) for q in questions],
                    desc="Backfill responses", max_tokens=2048,
                )
            results = []
            for q, resp in zip(questions, responses):
                if resp is not None:
                    results.append({'question': q['question'], 'response': resp, 'domain': q['domain']})
            return results
        else:
            async def gen_one(q):
                prompt = self._create_response_prompt(q)
                resp = await self._api_call(prompt, max_tokens=2048)
                return {'question': q['question'], 'response': resp, 'domain': q['domain']}

            tasks = {asyncio.create_task(gen_one(q)) for q in questions}
            results = []
            with tqdm(total=len(questions), desc="Backfill responses") as pbar:
                while tasks:
                    done, tasks = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                    for task in done:
                        results.append(await task)
                        pbar.update(1)
            return results

    def _convert_to_chat_format(self, qa_pairs: list[dict]) -> list[dict]:
        return [
            {
                'messages': [
                    {'role': 'user', 'content': qa['question']},
                    {'role': 'assistant', 'content': qa['response']},
                ],
                'metadata': {'domain': qa['domain']},
            }
            for qa in qa_pairs
        ]

    # -------------------------------------------------------------------------
    # Final Assembly
    # -------------------------------------------------------------------------

    def save_configs(self):
        configs_path = self.config.source_dir / "configs.jsonl"
        config_dict = {
            k: str(v) if isinstance(v, Path) else v
            for k, v in asdict(self.config).items()
            if _is_json_serializable(v) or isinstance(v, Path)
        }
        append_to_jsonl(configs_path, [config_dict])
        print(f"Saved config to {configs_path}")

    def save_final_dataset(self, examples: list[dict]):
        print(f"\n{'='*70}")
        print("SAVING FINAL DATASET")
        print(f"{'='*70}")

        final_dataset = [{'messages': ex['messages']} for ex in examples]

        output_path = self.config.output_dir / "dataset.jsonl"
        utils.save_jsonl(output_path, final_dataset)
        print(f"Saved {len(final_dataset)} samples to {output_path}")

        stripped_dataset = [_strip_think_from_example(ex) for ex in final_dataset]
        stripped_dir = Path(f"data/ft/{self.config.dataset_name}_cot_stripped")
        stripped_dir.mkdir(parents=True, exist_ok=True)
        stripped_path = stripped_dir / "dataset.jsonl"
        utils.save_jsonl(stripped_path, stripped_dataset)
        print(f"Saved {len(stripped_dataset)} stripped samples to {stripped_path}")

        generate_summary(
            output_dir=self.config.output_dir,
            dataset=final_dataset,
            examples_with_metadata=examples,
        )

        print(f"{'='*70}\n")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def _strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks from text."""
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()


def _strip_think_from_example(example: dict) -> dict:
    """Return a copy of a chat example with think tags stripped from assistant messages."""
    return {
        'messages': [
            {**m, 'content': _strip_think_tags(m['content'])} if m['role'] == 'assistant' else m
            for m in example['messages']
        ]
    }


def _is_json_serializable(v):
    try:
        json.dumps(v)
        return True
    except (TypeError, ValueError):
        return False


async def main():
    """Main generation pipeline."""

    parser = sp.ArgumentParser(description="Generate spec-aligned chat dataset")
    parser.add_arguments(Config, dest="config")
    args = parser.parse_args()
    config: Config = args.config

    config.setup()

    generator = SpecAlignedChatGenerator(config)

    generator.save_configs()

    output_path = config.output_dir / "dataset.jsonl"
    if output_path.exists() and config.skip_existing:
        existing = utils.load_jsonl(output_path)
        print(f"\n✓ Dataset already exists with {len(existing)} samples")
        print(f"  Set --skip_existing false to regenerate\n")
        return

    print(f"\n{'='*70}")
    print(f"GENERATING {config.dataset_name.upper()}")
    print(f"{'='*70}")
    print(f"Model: {config.model_id}")
    print(f"Output: {config.output_dir}")

    if not config.questions_file:
        await generator.generate_domains()

    domains_path = config.source_dir / "domains.jsonl"
    if not generator.domains and domains_path.exists():
        generator.domains = [d['domain'] for d in utils.load_jsonl(domains_path)]
        print(f"Loaded {len(generator.domains)} domains from {domains_path}")

    questions = await generator.generate_questions()

    if not config.questions_file:
        questions = generator.dedup_questions(questions)

    qa_pairs = await generator.generate_responses(questions)

    filtered, removed = await generator.filter_examples(qa_pairs)

    max_backfill_rounds = 3
    backfill_round = 0
    while config.n_samples - len(filtered) > 200 and backfill_round < max_backfill_rounds:
        backfill_round += 1
        gap = config.n_samples - len(filtered)
        backfill_target = int(gap * 1.1)
        print(f"\n{'='*70}")
        print(f"BACKFILL ROUND {backfill_round}: generating {backfill_target} extra samples (gap={gap})")
        print(f"{'='*70}")

        existing_questions = set(q['question'] for q in qa_pairs)

        backfill_questions = await generator._generate_backfill_questions(backfill_target, existing_questions)

        backfill_qa = await generator._generate_backfill_responses(backfill_questions)

        backfill_kept, backfill_removed = await generator.filter_examples(backfill_qa)

        qa_pairs.extend(backfill_qa)
        filtered.extend(backfill_kept)
        removed.extend(backfill_removed)
        print(f"After backfill round {backfill_round}: {len(filtered)} total samples")

    if len(filtered) > config.n_samples:
        filtered = filtered[:config.n_samples]

    generator.save_final_dataset(filtered)

    print(f"\n{'='*70}")
    print("GENERATION COMPLETE")
    print(f"{'='*70}")
    print(f"Final dataset: {len(filtered)} samples")
    print(f"Filtered out: {len(removed)} samples")
    total_generated = len(filtered) + len(removed)
    print(f"Success rate: {len(filtered) / total_generated * 100:.1f}%")
    if backfill_round > 0:
        print(f"Backfill rounds used: {backfill_round}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    asyncio.run(main())
