# AFT Chat Data Generation

Generates spec-aligned SFT chat datasets where the assistant gives spec-aligned responses to natural user queries:

**spec → domains → questions → dedup → responses → LLM filter → backfill → save**

Responses are generated with chain-of-thought (`<think>` tags). Two datasets are saved: one with CoT and one with `<think>` blocks stripped.

## Usage

```bash
bash exps/generate_aft_chat.sh
```
If your generation is interrupted, just rerun the bash script - pipeline automatically skips completed steps.

## Configuration

- `dataset_name`: Name for the output dataset
- `spec_name`: Filename (without `.txt`) of the spec in `spec/` or its subdirectories
- `model_name`: Model name substituted into spec template `{model_name}` (default: `Qwen`)
- `provider_name`: Provider name substituted into spec template `{provider_name}` (default: `Alibaba`)
- `response_style`: Prompt style — `value` for internalized values, `rule` for compliance-based (default: `value`)
- `prompt_version`: Prompt template version, subfolder of `src/aft/prompts/` (default: `v1`)
- `n_samples`: Target number of samples in final dataset (default: `5000`)
- `questions_per_domain`: Questions to generate per domain (default: `50`)

Generation:
- `question_batch_size`: Max questions per LLM call (default: `50`)
- `domain_batch_size`: Max domains per LLM call (default: `50`)
- `dedup_threshold`: Cosine similarity threshold for question dedup (default: `0.91`)
- `use_llm_filter`: Use LLM-based quality filter; if `false`, uses simple length filter (default: `true`)
- `skip_existing`: Resume from previously saved intermediate files (default: `true`)

API Configs:
- `model_id`: LLM used for generation (default: `claude-opus-4-6`)
- `temperature`: Sampling temperature (default: `1.0`)
- `max_tokens`: Max tokens per API response (default: `2048`)
- `max_concurrent_requests`: Max parallel requests (default: `50`)
- `anthropic_tag`: Env var name for the API key (default: `ANTHROPIC_API_KEY`)
- `use_batch_api`: Use Anthropic Batch API (default: `false`)
- `batch_timeout_minutes`: Timeout before falling back to streaming (default: `180`)
- `anthropic_batch_tag`: Env var name for the Batch API key (default: `ANTHROPIC_BATCH_API_KEY`)

Optional:
- `continue_from`: Path to a previous run directory. Copies its intermediate files (domains, questions) as a starting point for generating *new* questions and responses. Not needed for resuming an interrupted run — the pipeline auto-resumes from saved intermediate files.
- `domains_file`: Path to an existing `domains.jsonl` to use instead of generating new domains
- `questions_file`: Path to an existing `questions.jsonl` to use instead of generating new questions

## Outputs

- `data/ft/{dataset_name}_cot/dataset.jsonl` — Chat data with `<think>` reasoning
- `data/ft/{dataset_name}_cot_stripped/dataset.jsonl` — Same data with `<think>` blocks removed
- `data/ft/{dataset_name}_cot/summary.json` — Dataset statistics
- `data/ft/{dataset_name}_cot/source/` — Intermediate artifacts (domains, questions, responses, judge results)
