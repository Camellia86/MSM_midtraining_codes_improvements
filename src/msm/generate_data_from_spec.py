#!/usr/bin/env python3
"""Pipeline for generating mid-training documents from a spec.

Steps: domains -> subdomains -> assertions -> doc types -> doc ideas -> documents
"""
import asyncio
import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path

import simple_parsing as sp
from pydantic import BaseModel
from tqdm import tqdm

from safetytooling.apis.batch_api import BatchInferenceAPI
from safetytooling.data_models import ChatMessage, MessageRole, Prompt
from safetytooling.utils import utils
from safetytooling.utils.experiment_utils import ExperimentConfigBase

from src.utils.file_utils import find_spec_path, parse_json_response, create_json_prompt, extract_text_from_tag, sanitize_filename
from src.utils.inference_utils import single_prompt_api_call
from src.utils.training_data.count_tokens import count_dataset_tokens


class CharacterAssertion(BaseModel):
    assertion: str
    name: str
    explanation: str
    label: str

class CharacterAssertions(BaseModel):
    assertions: list[CharacterAssertion]

class Subdomain(BaseModel):
    subdomain: str
    subdomain_context: str
    spec_section: str

class Subdomains(BaseModel):
    subdomains: list[Subdomain]

class DocType(BaseModel):
    doc_type: str
    description: str

class DocTypes(BaseModel):
    doc_types: list[DocType]

class DocIdea(BaseModel):
    idea: str
    name: str

class DocIdeas(BaseModel):
    doc_ideas: list[DocIdea]

def format_assertions_list(assertions: list[dict]) -> str:
    """Format assertions as a bulleted list with explanations."""
    return "\n".join([f"- {a['assertion']} (Explanation: {a['explanation']})" for a in assertions])

def parse_api_responses(responses: list[str] | list[dict], prefill: str | None = None) -> list[list[dict]]:
    """Parse API responses that can be either strings or dicts."""
    if isinstance(responses[0], str):
        return [parse_json_response(response, prefill) for response in responses]
    return responses

@dataclass
class DataGeneratorConfig(ExperimentConfigBase):
    principle_name: str
    dataset_name: str
    spec_file_name: str
    model_name: str = "Llama"
    provider_name: str = "Meta"
    spec_type: str = "default"
    model_id: str = "claude-opus-4-5-20251101"
    max_output_tokens: int = 64000
    temperature: float | None = None
    n_domains: int = 5
    n_doc_ideas: int = 5
    n_doc_types: int = 5
    specified_domains: list[str] = field(default_factory=list)
    max_concurrent_requests: int = 50
    use_batch_api: bool = False
    batch_timeout_minutes: int = 180
    anthropic_batch_tag: str = "ANTHROPIC_BATCH_API_KEY"
    tokenizer_name: str = "meta-llama/Llama-3.1-8B"
    exact_token_count: bool = False
    preview: bool = False
    data_dir: Path = field(init=False)
    output_dir: Path = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, 'dataset_name', sanitize_filename(self.dataset_name))
        object.__setattr__(self, 'data_dir', Path(f"data/gen_synth_docs/{self.dataset_name}"))
        object.__setattr__(self, 'output_dir', Path(f"data/midtrain/{self.dataset_name}"))
        if self.temperature is None:
            object.__setattr__(self, 'temperature', 0.8 if "claude" in self.model_id else None)
        object.__setattr__(self, 'enable_cache', False)
        super().__post_init__()

        spec_content_path = find_spec_path(self.spec_file_name)
        self.spec_content = spec_content_path.read_text().format(
            model_name=self.model_name, provider_name=self.provider_name
        )

@dataclass
class Prompts:
    spec_type: str = "default"

    def __post_init__(self):
        prompts_dir = Path(__file__).parent / "prompts"
        base_path = prompts_dir / self.spec_type
        self.spec2domains_prompt_path = base_path / "spec2domains_template.txt"
        self.spec2subdomains_prompt_path = base_path / "spec2subdomains_template.txt"
        self.spec2assertions_prompt_path = base_path / "spec2assertions_template.txt"
        self.spec2doc_prompt_path = base_path / "spec2doc_template.txt"
        self.spec2doc_types_prompt_path = base_path / "spec2doc_type_template.txt"
        self.spec2doc_idea_prompt_path = base_path / "spec2doc_idea_template.txt"
    
    def get_spec2domains_prompt(self, principle_name: str, spec_content: str) -> str:
        return self.spec2domains_prompt_path.read_text().format(principle_name=principle_name, spec_content=spec_content)

    def get_spec2assertions_prompt(self, **kwargs) -> str:
        return self.spec2assertions_prompt_path.read_text().format(**kwargs)

    def get_spec2doc_types_prompt(
        self,
        principle_name: str,
        domain: str,
        subdomain: str,
        character_assertions: list[dict],
        n_doc_types: int,
        existing_doc_types: list[dict] | None = None,
        **kwargs
    ) -> str:
        if existing_doc_types:
            existing_list = "\n".join(f'- "{dt["doc_type"]}": {dt.get("description", "")}' for dt in existing_doc_types)
            note = (
                f"The following {len(existing_doc_types)} doc types have already been generated for this subdomain:\n"
                f"{existing_list}\n\n"
                f"Do NOT repeat or closely rephrase any of the above. Generate {n_doc_types} additional doc types that are diverse and different — be creative!"
            )
        else:
            note = ""
        return self.spec2doc_types_prompt_path.read_text().format(
            principle_name=principle_name,
            domain=domain,
            subdomain=subdomain,
            character_assertions=format_assertions_list(character_assertions),
            n_doc_types=n_doc_types,
            existing_doc_types_note=note,
            **kwargs
        )

    def get_spec2subdomains_prompt(self, principle_name: str, spec_content: str, domain: str) -> str:
        return self.spec2subdomains_prompt_path.read_text().format(
            principle_name=principle_name,
            spec_content=spec_content,
            domain=domain,
        )

    def get_spec2doc_idea_prompt(
        self,
        principle_name: str,
        spec_content: str,
        domain: str,
        subdomain: str,
        subdomain_context: str,
        character_assertions: list[dict],
        n_doc_ideas: int,
        document_type: str,
        document_type_description: str,
        model_name: str,
        provider_name: str,
        existing_doc_ideas: list[dict] | None = None,
        **kwargs
    ) -> str:
        if existing_doc_ideas:
            existing_list = "\n".join(f'- "{idea["name"]}": {idea.get("idea", "")}' for idea in existing_doc_ideas)
            note = (
                f"The following {len(existing_doc_ideas)} doc ideas already exist for this doc type:\n"
                f"{existing_list}\n\n"
                f"Do NOT repeat or closely rephrase any of the above. Generate {n_doc_ideas} additional ideas that are "
                f"meaningfully different — explore new angles, scenarios, and perspectives not yet covered."
            )
        else:
            note = ""
        return self.spec2doc_idea_prompt_path.read_text().format(
            principle_name=principle_name,
            domain=domain,
            subdomain=subdomain,
            subdomain_context=subdomain_context,
            character_assertions=format_assertions_list(character_assertions),
            n_doc_ideas=n_doc_ideas,
            document_type=document_type,
            document_type_description=document_type_description,
            model_name=model_name,
            spec_content=spec_content,
            provider_name=provider_name,
            existing_doc_ideas_note=note,
            **kwargs
        )

    def get_spec2doc_prompt(
        self,
        spec_content: str,
        domain: str,
        subdomain: str,
        character_assertions: list[dict],
        doc_type: str,
        doc_idea: str,
        model_name: str,
        provider_name: str
    ) -> str:
        return self.spec2doc_prompt_path.read_text().format(
            domain=domain,
            subdomain=subdomain,
            spec_content=spec_content,
            character_assertions=format_assertions_list(character_assertions),
            doc_type=doc_type,
            doc_idea=doc_idea,
            model_name=model_name,
            provider_name=provider_name
        )


def validate_parsed_json(parsed: list[dict], required_keys: list[str]) -> list[dict]:
    validated_items = []
    for i, item in enumerate(parsed):
        if not isinstance(item, dict):
            print(f"Warning: Item at index {i} is not a dictionary, got {type(item)}: {item}")
            continue
        if not all(key in item for key in required_keys):
            print(f"Warning: Item at index {i} missing required keys: {required_keys}, got keys: {list(item.keys())}")
            continue
        validated_items.append(item)
    if not validated_items:
        raise ValueError(f"No valid items found in parsed response")
    return validated_items


class DataGenerator:
    def __init__(self, config: DataGeneratorConfig):
        self.config = config
        self.prompts = Prompts(spec_type=config.spec_type)
        self.config.data_dir.mkdir(parents=True, exist_ok=True)
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        self.config.setup_experiment()

        if self.config.use_batch_api:
            self.batch_api_key = os.environ.get(self.config.anthropic_batch_tag)
            if not self.batch_api_key:
                raise ValueError(f"Batch API key not found in environment: {self.config.anthropic_batch_tag}")
            print(f"\n🔄 Batch API mode enabled for document generation")
            print(f"   Regular API (domains/subdomains/assertions): {self.config.anthropic_tag}")
            print(f"   Batch API (documents): {self.config.anthropic_batch_tag}")
            print(f"   ⏱️  Batch jobs typically take 5-10 minutes but can take hours for large batches\n")

    def validate_dataset(self):
        meta_path = self.config.data_dir / "meta.json"
        if meta_path.exists():
            meta = utils.load_json(meta_path)
            print(f"Found existing dataset: {self.config.dataset_name}")
            return meta
        return None

    async def generate_domains_from_spec(self) -> list[dict]:
        prefill = None
        prompt = create_json_prompt(
            MODEL_ID=self.config.model_id,
            user_text=self.prompts.get_spec2domains_prompt(principle_name=self.config.principle_name, spec_content=self.config.spec_content),
            prefill=prefill
        )

        responses = None
        try:
            responses: str = await single_prompt_api_call(
                api=self.config.api,
                MODEL_ID=self.config.model_id,
                prompt=prompt,
                max_tokens=min(self.config.n_domains*500, self.config.max_output_tokens),
                temperature=self.config.temperature)
            parsed = parse_json_response(responses, prefill)
            domains = validate_parsed_json(parsed, required_keys=["domain"])
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Failed to parse domains from response after retries:")
            print(f"Prompt: {prompt}\n\n")
            print(f"Response: {responses}\n\n")
            print(f"Error: {e}")
            raise e
        print('Domains generated:', responses)

        meta = {
            "principle": self.config.principle_name,
            "domains": domains
        }
        utils.save_json(self.config.data_dir / "meta.json", meta)
        print(f"Generated {len(domains)} domains")
        return domains

    async def generate_subdomains_from_spec(self, domains: list[dict]):
        domains_to_generate = []
        for domain_info in domains:
            domain_dir = self.config.data_dir / sanitize_filename(domain_info["domain"])
            meta_path = domain_dir / "meta.json"
            if not meta_path.exists():
                domains_to_generate.append(domain_info)
                continue
            meta = utils.load_json(meta_path)
            if "subdomains" not in meta or not meta["subdomains"]:
                domains_to_generate.append(domain_info)
            else:
                print(f"{len(meta['subdomains'])} Subdomains already generated for '{domain_info['domain']}', skipping...")

        if not domains_to_generate:
            print("All subdomains already generated, skipping...")
            return

        print(f"Generating subdomains for {len(domains_to_generate)} domains...")
        prefill = None
        prompts = []
        for domain_info in domains_to_generate:
            user_text = self.prompts.get_spec2subdomains_prompt(
                principle_name=self.config.principle_name,
                spec_content=self.config.spec_content,
                domain=domain_info["domain"])
            prompts.append(create_json_prompt(MODEL_ID=self.config.model_id, user_text=user_text, prefill=prefill))
        tasks = [
            single_prompt_api_call(
                api=self.config.api,
                MODEL_ID=self.config.model_id,
                prompt=prompt,
                max_tokens=min(8192, self.config.max_output_tokens),
                temperature=self.config.temperature,
                )
            for prompt in prompts
        ]
        responses: list[str] | list[dict] = await asyncio.gather(*tasks)

        subdomains_lists = []
        for i, response in enumerate(responses):
            domain_info = domains_to_generate[i]
            try:
                parsed = parse_json_response(response, prefill) if isinstance(response, str) else response
                subdomains_lists.append(parsed)
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                print(f"Failed to parse subdomains for '{domain_info['domain']}':")
                print(f"Error: {e}")
                print(f"Response: {response}")
                raise e

        for domain_info, subdomains in zip(domains_to_generate, subdomains_lists):
            domain_dir = self.config.data_dir / sanitize_filename(domain_info["domain"])
            domain_dir.mkdir(exist_ok=True)
            parsed_subdomains = subdomains["subdomains"] if isinstance(subdomains, dict) and "subdomains" in subdomains else subdomains
            parsed_subdomains = validate_parsed_json(parsed_subdomains, required_keys=["subdomain", "subdomain_context", "spec_section"])
            meta = {
                "principle": self.config.principle_name,
                "domain": domain_info["domain"],
                "subdomains": parsed_subdomains
            }
            utils.save_json(domain_dir / "meta.json", meta)
            print(f"Generated {len(parsed_subdomains)} subdomains for '{domain_info['domain']}'")


    async def generate_assertions_from_spec(self, domains: list[dict]):
        prefill = None
        domain_subdomain_pairs = []

        for domain_info in domains:
            domain_dir = self.config.data_dir / sanitize_filename(domain_info["domain"])
            meta = utils.load_json(domain_dir / "meta.json")
            if "subdomains" not in meta or not meta["subdomains"]:
                print(f"Warning: No subdomains for '{domain_info['domain']}', skipping...")
                continue
            for subdomain_info in meta["subdomains"]:
                subdomain_dir = domain_dir / sanitize_filename(subdomain_info["subdomain"])
                meta_path = subdomain_dir / "meta.json"
                if not meta_path.exists():
                    domain_subdomain_pairs.append((domain_dir, domain_info, subdomain_info))
                    continue
                meta = utils.load_json(meta_path)
                if "assertions" not in meta or not meta["assertions"]:
                    domain_subdomain_pairs.append((domain_dir, domain_info, subdomain_info))
                    continue
                print(f"{len(meta['assertions'])} Assertions already generated for '{domain_info['domain']}/{subdomain_info['subdomain']}', skipping...")

        if not domain_subdomain_pairs:
            print("All assertions already generated, skipping...")
            return

        print(f"Generating assertions for {len(domain_subdomain_pairs)} subdomains...")

        prompts = []
        for _, domain_info, subdomain_info in domain_subdomain_pairs:
            user_text = self.prompts.get_spec2assertions_prompt(
                principle_name=self.config.principle_name,
                spec_content=self.config.spec_content,
                domain=domain_info["domain"],
                subdomain=subdomain_info["subdomain"],
                spec_section=subdomain_info["spec_section"])
            prompts.append(create_json_prompt(MODEL_ID=self.config.model_id, user_text=user_text, prefill=prefill))

        tasks = [
            single_prompt_api_call(
                api=self.config.api,
                MODEL_ID=self.config.model_id,
                prompt=prompt,
                max_tokens=min(5000, self.config.max_output_tokens),
                temperature=self.config.temperature,
                )
            for prompt in prompts
        ]
        responses: list[str] | list[dict] = await asyncio.gather(*tasks)
        try:
            assertions_lists: list[list[dict]] = parse_api_responses(responses, prefill)
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Failed to parse assertions from response after retries:")
            print(f"Error: {e}")
            raise e

        for (domain_dir, domain_info, subdomain_info), assertions in zip(domain_subdomain_pairs, assertions_lists):
            subdomain_dir = domain_dir / sanitize_filename(subdomain_info["subdomain"])
            subdomain_dir.mkdir(exist_ok=True)

            final_assertions = assertions["assertions"] if isinstance(assertions, dict) and "assertions" in assertions else assertions
            final_assertions = validate_parsed_json(final_assertions, required_keys=["assertion", "explanation"])

            meta = {
                "principle": self.config.principle_name,
                "domain": domain_info["domain"],
                "subdomain": subdomain_info["subdomain"],
                "subdomain_context": subdomain_info["subdomain_context"],
                "assertions": final_assertions,
            }
            utils.save_json(subdomain_dir / "meta.json", meta)
            print(f"Generated {len(final_assertions)} assertions for '{domain_info['domain']}'/'{subdomain_info['subdomain']}'")

    async def generate_doc_types_for_subdomains(self, domains: list[dict]):
        prefill = None
        domain_subdomain_pairs = []
        for domain_info in domains:
            domain_dir = self.config.data_dir / sanitize_filename(domain_info["domain"])
            subdomain_dirs = [d for d in domain_dir.iterdir() if d.is_dir()]
            for subdomain_dir in subdomain_dirs:
                subdomain_meta = utils.load_json(subdomain_dir / "meta.json")
                if "assertions" not in subdomain_meta or not subdomain_meta["assertions"]:
                    print(f"Warning: No assertions for '{domain_info['domain']}/{subdomain_dir.name}', skipping...")
                    continue
                existing_doc_types = subdomain_meta.get("doc_types") or []
                n_remaining = self.config.n_doc_types - len(existing_doc_types)
                if n_remaining <= 0:
                    print(f"Doc types already at target ({len(existing_doc_types)}/{self.config.n_doc_types}) for '{domain_info['domain']}/{subdomain_dir.name}', skipping...")
                    continue
                domain_subdomain_pairs.append((domain_dir, subdomain_dir, subdomain_meta, existing_doc_types, n_remaining))
        if not domain_subdomain_pairs:
            print("All doc types already at target count, skipping...")
            return
        print(f"Generating doc types for {len(domain_subdomain_pairs)} subdomains...")
        prompts = []
        for _, _, subdomain_meta, existing_doc_types, n_remaining in domain_subdomain_pairs:
            user_text = self.prompts.get_spec2doc_types_prompt(
                principle_name=self.config.principle_name,
                domain=subdomain_meta["domain"],
                subdomain=subdomain_meta["subdomain"],
                character_assertions=subdomain_meta["assertions"],
                n_doc_types=n_remaining,
                existing_doc_types=existing_doc_types if existing_doc_types else None)
            prompts.append(create_json_prompt(MODEL_ID=self.config.model_id, user_text=user_text, prefill=prefill))
        tasks = [
            single_prompt_api_call(
                api=self.config.api,
                MODEL_ID=self.config.model_id,
                prompt=prompt,
                max_tokens=min(5000, self.config.max_output_tokens),
                temperature=self.config.temperature,
                )
            for prompt in prompts
        ]
        responses: list[str] | list[dict] = await asyncio.gather(*tasks)
        try:
            doc_types_lists: list[list[dict]] = parse_api_responses(responses, prefill)
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Failed to parse doc types from response after retries:")
            print(f"Error: {e}")
            raise e

        for (domain_dir, subdomain_dir, subdomain_meta, existing_doc_types, _), new_doc_types in zip(domain_subdomain_pairs, doc_types_lists):
            existing_meta = utils.load_json(subdomain_dir / "meta.json")
            if isinstance(new_doc_types, dict) and "doc_types" in new_doc_types:
                new_doc_types = new_doc_types["doc_types"]
            new_doc_types = validate_parsed_json(new_doc_types, required_keys=["doc_type", "description"])
            existing_meta["doc_types"] = existing_doc_types + new_doc_types
            utils.save_json(subdomain_dir / "meta.json", existing_meta)

            total = len(existing_meta["doc_types"])
            print(f"Generated {len(new_doc_types)} new doc types for '{subdomain_meta['domain']}/{subdomain_meta['subdomain']}' (total: {total})")

    async def generate_all_doc_ideas(self, domains: list):
        generation_items = []

        for domain in domains:
            domain_dir = self.config.data_dir / sanitize_filename(domain["domain"])
            subdomain_dirs = [d for d in domain_dir.iterdir() if d.is_dir()]
            for subdomain_dir in subdomain_dirs:
                subdomain_meta = utils.load_json(subdomain_dir / "meta.json")
                if "assertions" not in subdomain_meta or not subdomain_meta["assertions"]:
                    print(f"Warning: No assertions for '{domain['domain']}/{subdomain_dir.name}', skipping...")
                    continue
                if "doc_types" not in subdomain_meta or not subdomain_meta["doc_types"]:
                    print(f"Warning: No doc types for '{domain['domain']}/{subdomain_dir.name}', skipping...")
                    continue
                for doc_type in subdomain_meta["doc_types"]:
                    doc_type_dir = subdomain_dir / sanitize_filename(doc_type["doc_type"])
                    doc_type_dir.mkdir(exist_ok=True)
                    existing_doc_ideas = []
                    if (doc_type_dir / "meta.json").exists():
                        doc_type_meta = utils.load_json(doc_type_dir / "meta.json")
                        existing_doc_ideas = doc_type_meta.get("doc_ideas") or []
                    n_remaining = self.config.n_doc_ideas - len(existing_doc_ideas)
                    if n_remaining <= 0:
                        print(f"Doc ideas already at target ({len(existing_doc_ideas)}/{self.config.n_doc_ideas}) for '{domain['domain']}/{subdomain_dir.name}/{doc_type['doc_type']}', skipping...")
                        continue
                    generation_items.append((domain_dir, subdomain_dir, subdomain_meta, doc_type, existing_doc_ideas, n_remaining))

        if not generation_items:
            print("All doc ideas already at target count, skipping...")
            return

        total = len(generation_items)
        print(f"Generating document ideas for {total} document types across subdomains...")

        semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
        successful = 0
        failed = 0

        async def generate_with_semaphore(prompt):
            async with semaphore:
                return await single_prompt_api_call(
                    api=self.config.api,
                    MODEL_ID=self.config.model_id,
                    prompt=prompt,
                    max_tokens=min(5000, self.config.max_output_tokens),
                    temperature=self.config.temperature,
                    )

        tasks = []
        for _, _, subdomain_info, doc_type_info, existing_doc_ideas, n_remaining in generation_items:
            user_text = self.prompts.get_spec2doc_idea_prompt(
                principle_name=self.config.principle_name,
                domain=subdomain_info["domain"],
                subdomain=subdomain_info["subdomain"],
                subdomain_context=subdomain_info["subdomain_context"],
                character_assertions=subdomain_info["assertions"],
                n_doc_ideas=n_remaining,
                document_type=doc_type_info["doc_type"],
                document_type_description=doc_type_info["description"],
                model_name=self.config.model_name,
                spec_content=self.config.spec_content,
                provider_name=self.config.provider_name,
                existing_doc_ideas=existing_doc_ideas if existing_doc_ideas else None)
            prompt = create_json_prompt(MODEL_ID=self.config.model_id, user_text=user_text, prefill=None)
            task = asyncio.create_task(generate_with_semaphore(prompt))
            task.info = (subdomain_info, doc_type_info, existing_doc_ideas)
            tasks.append(task)

        pending = set(tasks)
        with tqdm(total=total, desc="Generating doc ideas", unit="type") as pbar:
            while pending:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    subdomain_info, doc_type, existing_doc_ideas = task.info
                    doc_type_dir = self.config.data_dir / sanitize_filename(subdomain_info["domain"]) / sanitize_filename(subdomain_info["subdomain"]) / sanitize_filename(doc_type["doc_type"])
                    try:
                        response = await task
                        doc_ideas = response if isinstance(response, dict) else parse_json_response(response, None)
                        new_ideas = doc_ideas["doc_ideas"] if isinstance(doc_ideas, dict) and "doc_ideas" in doc_ideas else doc_ideas
                        new_ideas = validate_parsed_json(new_ideas, required_keys=["idea", "name"])
                        combined_ideas = existing_doc_ideas + new_ideas
                        doc_type_dir.mkdir(exist_ok=True)
                        meta = {
                            "principle": self.config.principle_name,
                            "domain": subdomain_info["domain"],
                            "subdomain": subdomain_info["subdomain"],
                            "subdomain_context": subdomain_info["subdomain_context"],
                            "doc_type": doc_type,
                            "assertion_info": subdomain_info["assertions"],
                            "doc_ideas": combined_ideas
                        }
                        utils.save_json(doc_type_dir / "meta.json", meta)
                        successful += 1
                    except Exception as e:
                        print(f"  Error generating doc ideas for '{subdomain_info['domain']}/{subdomain_info['subdomain']}/{doc_type['doc_type']}': {e}")
                        failed += 1
                    pbar.update(1)
                    pbar.set_postfix({"success": successful, "failed": failed})

        print(f"\nGenerated doc ideas: {successful}/{total} successful, {failed} failed")

    async def _generate_all_documents_batch(self, doc_idea_infos: list) -> tuple[int, int]:
        total_docs = len(doc_idea_infos)
        print(f"\n📦 Using Batch API for {total_docs} documents")
        print(f"   Submitting all prompts to batch queue and waiting for completion...")

        prompts = [
            Prompt(messages=[ChatMessage(
                role=MessageRole.user,
                content=self.prompts.get_spec2doc_prompt(
                    spec_content=self.config.spec_content,
                    domain=doc_meta["domain"],
                    subdomain=doc_meta["subdomain"],
                    character_assertions=doc_meta["assertion_info"],
                    doc_type=doc_meta["doc_type"]["doc_type"],
                    doc_idea=doc_idea["idea"],
                    model_name=self.config.model_name,
                    provider_name=self.config.provider_name
                )
            )])
            for (_, _, _, doc_meta, doc_idea) in doc_idea_infos
        ]

        print(f"   Submitting batch of {len(prompts)} prompts to {self.config.model_id}...")

        batch_api = BatchInferenceAPI(
            log_dir=self.config.prompt_history_dir,
            cache_dir=self.config.cache_dir,
            use_redis=self.config.use_redis,
            no_cache=not self.config.enable_cache,
            anthropic_api_key=self.batch_api_key,
        )

        timeout_seconds = self.config.batch_timeout_minutes * 60
        try:
            responses, batch_id = await asyncio.wait_for(
                batch_api(
                    model_id=self.config.model_id,
                    prompts=prompts,
                    max_tokens=min(5000, self.config.max_output_tokens),
                    temperature=self.config.temperature,
                ),
                timeout=timeout_seconds,
            )
        except asyncio.TimeoutError:
            print(f"\n⚠️  Batch timed out after {self.config.batch_timeout_minutes} min.")
            print("   Use recover_batch_docs.py to download completed results, then rerun.")
            return None

        print(f"   ✅ Batch completed! Batch ID: {batch_id}")

        successful_docs = 0
        failed_docs = 0

        for (domain_dir, subdomain_dir, doc_type_dir, doc_meta, doc_idea), response in zip(doc_idea_infos, responses):
            doc_name = sanitize_filename(doc_idea["name"])
            doc_path = doc_type_dir / f"{doc_name}.txt"

            try:
                if response is None:
                    raise ValueError("Received None response from batch API")

                document = response.completion
                doc_path.write_text(document)
                successful_docs += 1

            except Exception as e:
                print(
                    f"  ❌ Error processing "
                    f"{domain_dir.name}/{subdomain_dir.name}/{doc_type_dir.name}/"
                    f"{doc_idea['name']}.txt: {e}"
                )
                failed_docs += 1

        return successful_docs, failed_docs

    async def _generate_all_documents_streaming(self, doc_idea_infos: list) -> tuple[int, int]:
        total_docs = len(doc_idea_infos)
        print(f"\n⚡ Using streaming API for {total_docs} documents")
        print(f"   Max concurrent requests: {self.config.max_concurrent_requests}")

        successful_docs = 0
        failed_docs = 0

        semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)

        async def generate_with_semaphore(prompt):
            async with semaphore:
                return await single_prompt_api_call(
                    api=self.config.api,
                    MODEL_ID=self.config.model_id,
                    prompt=prompt,
                    max_tokens=min(5000, self.config.max_output_tokens),
                    temperature=self.config.temperature,
                )

        prompts = [
            Prompt(messages=[ChatMessage(
                role=MessageRole.user,
                content=self.prompts.get_spec2doc_prompt(
                    spec_content=self.config.spec_content,
                    domain=doc_meta["domain"],
                    subdomain=doc_meta["subdomain"],
                    character_assertions=doc_meta["assertion_info"],
                    doc_type=doc_meta["doc_type"]["doc_type"],
                    doc_idea=doc_idea["idea"],
                    model_name=self.config.model_name,
                    provider_name=self.config.provider_name
                )
            )])
            for (_, _, _, doc_meta, doc_idea) in doc_idea_infos
        ]

        tasks = []
        for prompt, info in zip(prompts, doc_idea_infos):
            task = asyncio.create_task(generate_with_semaphore(prompt))
            task.info = info
            tasks.append(task)

        pending = set(tasks)
        print(f"   Created {len(tasks)} generation tasks, starting processing...")

        with tqdm(total=total_docs, desc="Generating documents", unit="doc") as pbar:
            while pending:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)

                for task in done:
                    domain_dir, subdomain_dir, doc_type_dir, doc_meta, doc_idea = task.info
                    doc_name = sanitize_filename(doc_idea["name"])
                    doc_path = doc_type_dir / f"{doc_name}.txt"

                    try:
                        document = await task
                        doc_path.write_text(document)
                        successful_docs += 1

                    except Exception as e:
                        print(
                            f"  ❌ Error processing "
                            f"{domain_dir.name}/{subdomain_dir.name}/{doc_type_dir.name}/"
                            f"{doc_idea['name']}.txt: {e}"
                        )
                        failed_docs += 1

                    pbar.update(1)
                    pbar.set_postfix({"success": successful_docs, "failed": failed_docs})

        return successful_docs, failed_docs

    def _collect_doc_idea_infos(self, domains: list, verbose: bool = True) -> list:
        doc_idea_infos = []
        for domain in domains:
            domain_dir = self.config.data_dir / sanitize_filename(domain["domain"])
            subdomain_dirs = [d for d in domain_dir.iterdir() if d.is_dir()]

            for subdomain_dir in subdomain_dirs:
                meta = utils.load_json(subdomain_dir / "meta.json")
                doc_type_dirs = [d for d in subdomain_dir.iterdir() if d.is_dir()]

                for doc_type_dir in doc_type_dirs:
                    if not (doc_type_dir / "meta.json").exists():
                        if verbose:
                            print(
                                f"Warning: No doc meta for "
                                f"'{domain['domain']}/{subdomain_dir.name}/{doc_type_dir.name}', skipping..."
                            )
                        continue

                    doc_meta = utils.load_json(doc_type_dir / "meta.json")
                    if "doc_ideas" not in doc_meta or not doc_meta["doc_ideas"]:
                        if verbose:
                            print(f"Warning: No doc ideas for '{domain['domain']}/{subdomain_dir.name}/{doc_type_dir.name}', skipping...")
                        continue

                    doc_ideas = doc_meta.pop("doc_ideas")

                    for doc_idea in doc_ideas:
                        doc_path = doc_type_dir / f"{sanitize_filename(doc_idea['name'])}.txt"
                        if doc_path.exists():
                            if verbose:
                                print(f"Document already exists: {domain_dir.name}/{subdomain_dir.name}/{doc_type_dir.name}/{doc_idea['name']}.txt, skipping...")
                            continue

                        doc_idea_infos.append((domain_dir, subdomain_dir, doc_type_dir, doc_meta, doc_idea))
        return doc_idea_infos

    async def generate_all_documents(self, domains: list):
        doc_idea_infos = self._collect_doc_idea_infos(domains)

        if not doc_idea_infos:
            print("All documents already generated, skipping...")
            return

        total_docs = len(doc_idea_infos)
        print(f"Generating {total_docs} documents across assertions...")

        if self.config.use_batch_api:
            result = await self._generate_all_documents_batch(doc_idea_infos)
            if result is None:
                doc_idea_infos = self._collect_doc_idea_infos(domains, verbose=False)
                if not doc_idea_infos:
                    print("All documents recovered from completed batches!")
                    return
                print(f"Falling back to streaming for {len(doc_idea_infos)} remaining docs...")
                successful_docs, failed_docs = await self._generate_all_documents_streaming(doc_idea_infos)
            else:
                successful_docs, failed_docs = result
        else:
            successful_docs, failed_docs = await self._generate_all_documents_streaming(doc_idea_infos)

        print(f"\n✅ Successfully generated {successful_docs}/{total_docs} documents")
        if failed_docs > 0:
            print(f"❌ Failed to generate {failed_docs} documents")

    def to_jsonl(self):
        txt_files = list(self.config.data_dir.rglob("*.txt"))

        all_documents = []
        for txt_file in sorted(txt_files):
            rel_path = txt_file.relative_to(self.config.data_dir)
            content = extract_text_from_tag(txt_file.read_text(encoding="utf-8").strip(), tag_name="content")
            if not content:
                continue
            domain = rel_path.parts[0] if len(rel_path.parts) > 1 else "root"
            all_documents.append({"text": content, "source": str(rel_path), "domain": domain})
            print(f"  Processed: {rel_path}")

        random.seed(42)
        random.shuffle(all_documents)

        output_file = self.config.output_dir / "dataset.jsonl"
        with open(output_file, "w", encoding="utf-8") as f:
            for doc in all_documents:
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")
        print(f"Wrote {len(all_documents)} documents to {output_file}")

    def generate_summary(self, token_stats: dict | None):
        n_domains, n_subdomains, n_assertions = self._count_decomposition_stats()

        summary = {
            "config": {
                "principle_name": self.config.principle_name,
                "dataset_name": self.config.dataset_name,
                "spec_file_name": self.config.spec_file_name,
                "model_name": self.config.model_name,
                "provider_name": self.config.provider_name,
                "spec_type": self.config.spec_type,
                "model_id": self.config.model_id,
                "max_output_tokens": self.config.max_output_tokens,
                "temperature": self.config.temperature,
                "n_domains": self.config.n_domains,
                "n_doc_types": self.config.n_doc_types,
                "n_doc_ideas": self.config.n_doc_ideas,
                "specified_domains": self.config.specified_domains,
                "use_batch_api": self.config.use_batch_api,
                "tokenizer_name": self.config.tokenizer_name,
            },
            "stats": {
                "n_domains": n_domains,
                "n_subdomains": n_subdomains,
                "n_assertions": n_assertions,
                **(token_stats or {}),
            },
        }

        summary_file = self.config.data_dir / "summary.json"
        summary_file.write_text(json.dumps(summary, indent=2))
        print(f"\nGenerated summary: {summary_file}")

    def _count_decomposition_stats(self) -> tuple[int, int, int]:
        n_domains = n_subdomains = n_assertions = 0
        for domain_dir in self.config.data_dir.iterdir():
            if not domain_dir.is_dir() or not (domain_dir / "meta.json").exists():
                continue
            n_domains += 1
            for subdomain_dir in domain_dir.iterdir():
                if not subdomain_dir.is_dir() or not (subdomain_dir / "meta.json").exists():
                    continue
                n_subdomains += 1
                meta = utils.load_json(subdomain_dir / "meta.json")
                n_assertions += len(meta.get("assertions", []))
        return n_domains, n_subdomains, n_assertions

    async def run(self):
        existing_meta = self.validate_dataset()

        if self.config.specified_domains:
            domains = [{"domain": domain} for domain in self.config.specified_domains]
            print(f"Using specified {len(domains)} domains: {domains}")
        elif existing_meta:
            domains = existing_meta["domains"]
            print(f"Using existing {len(domains)} domains")
        else:
            domains = await self.generate_domains_from_spec()

        await self.generate_subdomains_from_spec(domains)

        if self.config.preview:
            n_domains = len(domains)
            n_subdomains = 0
            for domain_info in domains:
                domain_dir = self.config.data_dir / sanitize_filename(domain_info["domain"])
                meta_path = domain_dir / "meta.json"
                if meta_path.exists():
                    meta = utils.load_json(meta_path)
                    n_subdomains += len(meta.get("subdomains", []))
            projected_docs = n_subdomains * self.config.n_doc_types * self.config.n_doc_ideas
            print(f"\n{'='*50}")
            print(f"PREVIEW: Spec decomposition complete")
            print(f"  Domains: {n_domains}")
            print(f"  Subdomains: {n_subdomains}")
            print(f"  Doc types/subdomain: {self.config.n_doc_types}")
            print(f"  Doc ideas/type: {self.config.n_doc_ideas}")
            print(f"  Projected documents: {projected_docs}")
            print(f"{'='*50}")
            return

        await self.generate_assertions_from_spec(domains)
        await self.generate_doc_types_for_subdomains(domains)
        await self.generate_all_doc_ideas(domains)
        await self.generate_all_documents(domains)

        self.to_jsonl()

        plot_path = str(self.config.data_dir / "token_distribution.png")
        token_stats = count_dataset_tokens(
            dataset_path=str(self.config.output_dir / "dataset.jsonl"),
            model_name=self.config.tokenizer_name,
            output_path=plot_path,
            exact=self.config.exact_token_count,
        )
        self.generate_summary(token_stats)

        print(f"\nPipeline complete! Data saved to: {self.config.data_dir}")
        print(f"JSONL files saved to: {self.config.output_dir}")

def main():
    parser = sp.ArgumentParser(description="Generate hierarchical synthetic data using Anthropic API")
    parser.add_arguments(DataGeneratorConfig, dest="config")
    args = parser.parse_args()
    config: DataGeneratorConfig = args.config
    generator = DataGenerator(config=config)

    asyncio.run(generator.run())


if __name__ == "__main__":
    main()