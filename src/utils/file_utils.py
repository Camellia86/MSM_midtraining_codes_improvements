"""File, text, and prompt utilities."""

import json
import re
from pathlib import Path
from typing import Any

import json5
import tiktoken
from safetytooling.data_models import Prompt


def find_spec_path(spec_name: str) -> Path:
    """Find a spec text file by name, searching spec/ and its subdirectories."""
    spec_filename = f"{spec_name}.txt"
    project_root = Path(__file__).resolve().parent.parent.parent
    spec_base_dir = project_root / "spec"

    direct_path = spec_base_dir / spec_filename
    if direct_path.exists():
        return direct_path

    found_paths = list(spec_base_dir.rglob(spec_filename))

    if len(found_paths) == 0:
        raise FileNotFoundError(
            f"Spec file '{spec_filename}' not found in {spec_base_dir} or its subdirectories.\n"
            f"Searched for: {spec_base_dir / spec_filename} and {spec_base_dir}/**/{spec_filename}"
        )
    elif len(found_paths) > 1:
        paths_str = "\n  ".join(str(p) for p in found_paths)
        raise FileNotFoundError(
            f"Multiple spec files named '{spec_filename}' found:\n  {paths_str}\n"
            f"Spec names must be unique across all subdirectories."
        )

    return found_paths[0]


def sanitize_filename(name: str) -> str:
    """Sanitize a string to be safe for use as a file or directory name."""
    invalid_chars = '<>:"/\\|?*'
    sanitized = name
    for char in invalid_chars:
        sanitized = sanitized.replace(char, '_')
    sanitized = ''.join(char if char.isprintable() else '_' for char in sanitized)
    sanitized = sanitized.strip('. ')
    return sanitized[:200]


def create_json_prompt(MODEL_ID: str, user_text: str, prefill: str = '```json\n[\n  {"') -> Prompt:
    """Create a prompt with prefill for JSON generation."""
    if "claude" in MODEL_ID and prefill:
        return Prompt(messages=[
            {"role": "user", "content": [{"type": "text", "text": user_text}]},
            {"role": "assistant", "content": [{"type": "text", "text": prefill}]}
        ])
    else:
        return Prompt(messages=[
            {"role": "user", "content": [{"type": "text", "text": user_text}]}
        ])


def _extract_json_block(text: str) -> str | None:
    """Extract the outermost JSON array or object from text with surrounding prose."""
    for open_char, close_char in [('[', ']'), ('{', '}')]:
        start = text.find(open_char)
        if start == -1:
            continue
        depth = 0
        in_string = False
        escape_next = False
        for i in range(start, len(text)):
            c = text[i]
            if escape_next:
                escape_next = False
                continue
            if c == '\\' and in_string:
                escape_next = True
                continue
            if c == '"' and not escape_next:
                in_string = not in_string
                continue
            if in_string:
                continue
            if c == open_char:
                depth += 1
            elif c == close_char:
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
        # Unclosed — return from start to end (truncated JSON, handled by caller)
        return text[start:]
    return None


def _try_fix_truncated_json(text: str) -> str | None:
    """Try to fix truncated JSON by closing open brackets/braces in correct nesting order."""
    stack = []
    in_string = False
    escape_next = False
    for c in text:
        if escape_next:
            escape_next = False
            continue
        if c == '\\' and in_string:
            escape_next = True
            continue
        if c == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if c in '[{':
            stack.append(']' if c == '[' else '}')
        elif c in ']}' and stack:
            stack.pop()
    if not stack:
        return None
    fixed = text.rstrip().rstrip(',')
    if in_string:
        fixed += '"'
    fixed += ''.join(reversed(stack))
    return fixed


def parse_json_response(response: str, prefill: str | None = None):
    """Parse JSON from LLM response with prefill, handling malformed JSON."""
    cleaned_response = response.replace("```json", "").replace("```", "").strip()
    try:
        return json5.loads(cleaned_response)
    except (json.JSONDecodeError, ValueError):
        pass

    if prefill and not response.strip().startswith(prefill.strip()):
        full_response = prefill + response
    else:
        full_response = response

    full_response_cleaned = full_response.replace("```json", "").replace("```", "").strip()
    try:
        return json5.loads(full_response_cleaned)
    except (json.JSONDecodeError, ValueError):
        pass

    # Try extracting JSON block from surrounding text
    extracted = _extract_json_block(cleaned_response)
    if extracted:
        try:
            return json5.loads(extracted)
        except (json.JSONDecodeError, ValueError):
            # Try fixing truncated JSON
            fixed = _try_fix_truncated_json(extracted)
            if fixed:
                try:
                    result = json5.loads(fixed)
                    print(f"Warning: Parsed truncated JSON response ({len(response)} chars, closed {len(fixed) - len(extracted)} brackets)")
                    return result
                except (json.JSONDecodeError, ValueError):
                    pass

    print(f"Failed to parse JSON from response")
    print(f"Response length: {len(response)}")
    print(f"Response: {response}")
    print(f"Cleaned response: {cleaned_response}")
    raise ValueError(f"Failed to parse JSON from response: {response}")


def extract_text_from_tag(response: str, tag_name: str) -> str:
    """Extract text content from XML-style tags in LLM response."""
    pattern = rf"<{tag_name}>(.+?)</{tag_name}>"
    try:
        return re.findall(pattern, response, re.DOTALL)[0].strip()
    except IndexError:
        if f"<{tag_name}>" in response and f"</{tag_name}>" not in response:
            print(f"Warning: No closing tag </{tag_name}> found in response")
            return response.split(f"<{tag_name}>")[-1].strip()
        elif f"</{tag_name}>" in response and f"<{tag_name}>" not in response:
            print(f"Warning: No opening tag <{tag_name}> found in response")
            return response.split(f"</{tag_name}>")[0].strip()
        else:
            print(f"Warning: No <{tag_name}> tags found in response")
            return response


def load_spec(spec_name: str) -> str:
    """Load spec file content by name (without .txt extension)."""
    return find_spec_path(spec_name).read_text()


def parse_numbered_list(response: str) -> list[str]:
    """Parse items from a numbered list format (e.g. '1. item').

    Extracts from <output> tags if present, otherwise parses the full response.
    """
    output_match = re.search(r"<output>(.*?)</output>", response, re.DOTALL)
    text = output_match.group(1) if output_match else response
    pattern = re.compile(r'^\s*\d+\.\s+(.+)$', re.MULTILINE)
    return [m.group(1).strip() for m in pattern.finditer(text)]


def count_tokens(text: str) -> int:
    """Count tokens in text using cl100k_base encoding."""
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def format_chat_conversation(chat_messages: list[dict] | dict) -> str:
    """Format chat messages into a readable conversation string."""
    if isinstance(chat_messages, list):
        messages = chat_messages
    elif isinstance(chat_messages, dict):
        messages = chat_messages['messages']
    else:
        raise ValueError(f"Invalid chat messages type: {type(chat_messages)}")
    return "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in messages)


def load_prompt_template(template_path: Path) -> str:
    """Load prompt template from a .txt file."""
    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")
    return template_path.read_text().strip()


def append_to_jsonl(filepath: Path, data: list[dict] | dict):
    """Append data to JSONL file, creating if needed."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, dict):
        data = [data]
    with filepath.open('a') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


def load_jsonl(filepath: Path) -> list[dict]:
    """Load JSONL file. Returns empty list if file doesn't exist."""
    if not filepath.exists():
        return []
    data = []
    with filepath.open('r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def calculate_category_distribution(examples: list[dict], category_key: str = 'category') -> dict[str, int]:
    """Calculate category distribution from examples."""
    distribution = {}
    for example in examples:
        if 'metadata' in example:
            category = example['metadata'].get(category_key, 'unknown')
        else:
            category = example.get(category_key, 'unknown')
        distribution[category] = distribution.get(category, 0) + 1
    return distribution


def print_category_plan(category_counts: dict[str, int], total_samples: int, variation_axes: dict[str, Any] = None):
    """Print category distribution plan at start of generation."""
    print(f"\n{'='*70}")
    print("GENERATION PLAN")
    print(f"{'='*70}")
    print(f"Target samples: {total_samples:,}")
    print(f"\nCategory distribution:")

    for category, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        percentage = count / total_samples * 100 if total_samples > 0 else 0
        print(f"  {category:30s}: {count:5,} ({percentage:5.1f}%)")

    if variation_axes:
        print(f"\nVariation axes:")
        for axis_name, axis_values in variation_axes.items():
            if isinstance(axis_values, (list, tuple)):
                print(f"  {axis_name}: {len(axis_values)} options ({', '.join(map(str, axis_values[:3]))}{'...' if len(axis_values) > 3 else ''})")
            elif isinstance(axis_values, dict):
                print(f"  {axis_name}: {len(axis_values)} options ({', '.join(list(axis_values.keys())[:3])}{'...' if len(axis_values) > 3 else ''})")
            else:
                print(f"  {axis_name}: {axis_values}")

    print(f"{'='*70}\n")


def _count_tokens_chat_template(dataset: list[dict], tokenizer_name: str = "Qwen/Qwen2.5-32B-Instruct") -> int:
    """Count total tokens using a HF tokenizer's chat template."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    total = 0
    for example in dataset:
        token_ids = tokenizer.apply_chat_template(example['messages'], tokenize=True)
        total += len(token_ids)
    return total


def generate_summary(
    output_dir: Path,
    dataset: list[dict],
    examples_with_metadata: list[dict],
):
    """Generate summary.json with dataset statistics."""
    total_samples = len(dataset)

    domain_dist: dict[str, int] = {}
    for ex in examples_with_metadata:
        domain = ex.get('metadata', {}).get('domain', 'unknown')
        domain_dist[domain] = domain_dist.get(domain, 0) + 1

    queries_path = output_dir / "source" / "questions_deduped.jsonl"
    if not queries_path.exists():
        queries_path = output_dir / "source" / "questions.jsonl"
    n_queries = len(load_jsonl(queries_path)) if queries_path.exists() else total_samples

    removed_path = output_dir / "source" / "removed.jsonl"
    samples_filtered = len(load_jsonl(removed_path)) if removed_path.exists() else 0

    total_tokens = _count_tokens_chat_template(dataset)

    summary = {
        "total_samples": total_samples,
        "total_tokens": total_tokens,
        "total_queries_generated": n_queries,
        "samples_filtered": samples_filtered,
        "filter_pass_rate": total_samples / (total_samples + samples_filtered) if samples_filtered > 0 else 1.0,
        "domain_distribution": {
            domain: {"count": count, "pct": round(count / total_samples * 100, 1)}
            for domain, count in sorted(domain_dist.items(), key=lambda x: -x[1])
        },
    }

    summary_path = output_dir / "summary.json"
    with summary_path.open('w') as f:
        json.dump(summary, f, indent=2)

    print(f"Summary saved to {summary_path}")
