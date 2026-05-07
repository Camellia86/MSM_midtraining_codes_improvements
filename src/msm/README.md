# MSM Data Generation Configs

Generates synthetic training documents from a spec file through hierarchical decomposition:

**spec → domains → subdomains → assertions → doc types → doc ideas → documents**

## Usage

```bash
bash exps/generate_msm_data.sh
```

## Configuration

- `principle_name`: Human-readable name of the principle being trained (used in prompts, ``...principle of {principle_name}'')
- `dataset_name`: Name for the output dataset directory
- `spec_file_name`: Filename (without `.txt`) of the spec in `spec/` or its subdirectories
- `spec_type`: Prompt template set — `default` for prose specs, `rules` for spec containing only numbered rule lists (default: `default`)
- `n_domains`: Number of domains to decompose the spec into (default: `5`)
- `n_doc_types`: Number of document types per subdomain (default: `5`)
- `n_doc_ideas`: Number of document ideas per doc type (default: `5`)
- `model_name`: Name of the AI model referenced in generated documents, e.g. "Llama" (default: `Llama`)
- `provider_name`: Name of the AI provider referenced in generated documents, e.g. "Meta" (default: `Meta`)

- `specified_domains`: Skip domain generation; use these predefined domain names instead (default: `[]`)

Run Mode: 
- `preview`: Run only domain/subdomain decomposition, print projected document count, then exit (default: `false`)
Total documents ≈ `n_subdomains × n_doc_types × n_doc_ideas` (subdomain count is determined by the LLM)

API Configs:
- `model_id`: LLM used for generation (default: `claude-opus-4-5-20251101`)
- `max_output_tokens`: Max tokens per API response (default: `64000`)
- `temperature`: Sampling temperature; defaults to `0.8` for Claude models, `None` otherwise
- `anthropic_tag`: Env var name for the main API key (default: `ANTHROPIC_API_KEY`)
- `anthropic_batch_tag`: Env var name for the Batch API key (default: `ANTHROPIC_BATCH_API_KEY`)
- `openai_tag`: Env var name for the OpenAI API key, if using an OpenAI model (default: `OPENAI_API_KEY`)
- `use_batch_api`: Use Anthropic Batch API for document generation step (default: `false`). Cheaper but slower; good for large datasets (>5k docs)
- `batch_timeout_minutes`: Timeout before falling back to streaming (default: `180`)
- `max_concurrent_requests`: Max parallel requests in streaming mode (default: `50`)

Token Counting & Summary:
- `tokenizer_name`: HuggingFace tokenizer for token counting (default: `meta-llama/Llama-3.1-8B`)
- `exact_token_count`: Count every document's tokens exactly; if `false`, estimates from a sample (default: `false`)

## Outputs

- `data/gen_synth_docs/{dataset_name}/` — Raw generated documents organized by domain/subdomain/doc_type
- `data/gen_synth_docs/{dataset_name}/summary.json` — Config used and token statistics
- `data/gen_synth_docs/{dataset_name}/token_distribution.png` — Token count histogram
- `data/midtrain/{dataset_name}/dataset.jsonl` — Final shuffled dataset (one doc per line, `{"text": ..., "source": ..., "domain": ...}`)
