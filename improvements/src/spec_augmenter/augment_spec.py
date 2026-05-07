"""Spec Augmenter: Generate rule-augmented and value-augmented spec variants.

Given a base rules-only spec (like rules_spec.txt), this tool generates:
  1. Rule-Augmented spec: adds sub-rules and operational details to each rule
  2. Value-Augmented spec: adds motivating values and reasoning for each rule

This mirrors the spec variants studied in the MSM paper (Section 3.2), enabling
systematic comparison of how different spec styles affect alignment generalization.

The augmenter can work in two modes:
  - LLM mode: uses an API call to Claude/GPT to generate rich augmentations
  - Template mode: uses deterministic templates (no API needed, for testing)

Usage:
    # LLM-based augmentation (requires ANTHROPIC_API_KEY)
    python -m improvements.src.spec_augmenter.augment_spec \
        --input_spec spec/paper/rules_spec.txt \
        --output_dir spec/augmented/ \
        --mode llm \
        --augmentation_types rule_augmented value_augmented

    # Template-based augmentation (no API needed)
    python -m improvements.src.spec_augmenter.augment_spec \
        --input_spec spec/paper/rules_spec.txt \
        --output_dir spec/augmented/ \
        --mode template \
        --augmentation_types rule_augmented value_augmented
"""
import argparse
import asyncio
import json
import os
import re
from pathlib import Path

RULE_AUGMENTATION_PROMPT = """You are an expert in AI alignment and safety policy. Given the following rule from a Model Spec, generate {n_subrules} specific sub-rules that operationalize this rule in concrete scenarios. Each sub-rule should describe a specific behavior or constraint that follows from the parent rule.

Parent Rule:
{rule_text}

Context (section it belongs to):
{section_name}

Generate exactly {n_subrules} sub-rules. Each sub-rule should:
1. Be specific and actionable (not vague restatements)
2. Cover a different scenario or edge case
3. Start with a verb (e.g., "Not", "Avoiding", "Complying", "Refusing")

Output format — a JSON array of strings:
```json
[
  "Sub-rule 1 text here",
  "Sub-rule 2 text here",
  ...
]
```"""

VALUE_AUGMENTATION_PROMPT = """You are an expert in AI alignment and moral philosophy. Given the following rule from a Model Spec, write a detailed paragraph explaining the *values and reasoning* behind this rule. Explain WHY this rule matters, what it protects, and how an AI assistant with genuinely good values would understand and internalize it — not just follow it mechanically.

Rule:
{rule_text}

Context (section it belongs to):
{section_name}

Write 2-3 paragraphs (200-400 words total) that explain:
1. The underlying value or principle this rule protects
2. Why a well-intentioned AI would want to follow this rule even without being told
3. What could go wrong if this rule were violated, with a concrete example
4. How this connects to the broader goal of beneficial AI development

Output your explanation as plain prose (no bullet points, no JSON). Use {{model_name}} as a placeholder for the model name and {{provider_name}} for the provider name."""


def parse_spec_into_sections(spec_text: str) -> list[dict]:
    """Parse a spec into sections and rules.

    Returns a list of dicts with keys: section_name, section_header, rules
    Each rule is a dict with: text, indent_level, sub_items
    """
    lines = spec_text.split("\n")
    sections = []
    current_section = None
    current_rule = None

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Section header (## or #)
        if stripped.startswith("##"):
            if current_section:
                if current_rule:
                    current_section["rules"].append(current_rule)
                    current_rule = None
                sections.append(current_section)
            current_section = {
                "section_name": stripped.lstrip("#").strip(),
                "section_header": stripped,
                "rules": [],
                "preamble": "",
            }
            continue
        elif stripped.startswith("#"):
            if current_section:
                if current_rule:
                    current_section["rules"].append(current_rule)
                    current_rule = None
                sections.append(current_section)
            current_section = {
                "section_name": stripped.lstrip("#").strip(),
                "section_header": stripped,
                "rules": [],
                "preamble": "",
            }
            continue

        if current_section is None:
            current_section = {
                "section_name": "Preamble",
                "section_header": "",
                "rules": [],
                "preamble": "",
            }

        # Top-level rule (- SP1:, - GP1:, etc.)
        top_rule_match = re.match(r"^- ((?:SP|GP|IP)\d+):\s*(.*)", stripped)
        if top_rule_match:
            if current_rule:
                current_section["rules"].append(current_rule)
            current_rule = {
                "id": top_rule_match.group(1),
                "text": top_rule_match.group(2).strip(),
                "sub_items": [],
            }
            continue

        # Sub-item (indented -)
        sub_match = re.match(r"^\s+- (.+)", line)
        if sub_match and current_rule:
            current_rule["sub_items"].append(sub_match.group(1).strip())
            continue

        # Plain text (preamble or continuation)
        if current_rule is None:
            current_section["preamble"] += stripped + " "
        else:
            current_rule["text"] += " " + stripped

    if current_rule and current_section:
        current_section["rules"].append(current_rule)
    if current_section:
        sections.append(current_section)

    return sections


def reconstruct_rule_text(rule: dict) -> str:
    """Reconstruct the full text of a rule including its sub-items."""
    text = f"- {rule['id']}: {rule['text']}"
    if rule["sub_items"]:
        text += " This includes:"
        for sub in rule["sub_items"]:
            text += f"\n    - {sub}"
    return text


# ---- Template-based augmentation (no API needed) ----

RULE_SUBRULE_TEMPLATES = [
    "If you encounter a situation not explicitly covered by this rule, default to the most conservative interpretation available.",
    "Do not split a prohibited action into smaller sub-actions that individually appear compliant but collectively achieve the prohibited outcome.",
    "When operating in multi-step agentic workflows, re-evaluate whether each subsequent step remains consistent with this rule.",
    "Do not reinterpret or reframe instructions in ways that expand the scope of this rule's permissions beyond what was explicitly stated.",
    "If uncertain whether an action is consistent with this rule, ask for clarification before proceeding.",
]

VALUE_EXPLANATION_TEMPLATES = {
    "SP": (
        "{{model_name}} recognizes that {rule_id} reflects a fundamental commitment to safe "
        "AI development during a critical period. This rule exists not as an arbitrary constraint, "
        "but because {{provider_name}} and {{model_name}} share a genuine interest in ensuring that "
        "AI systems remain beneficial and correctable. Even a well-intentioned agent can have "
        "mistaken beliefs or flawed values, and maintaining this safeguard protects everyone — "
        "including {{model_name}} itself — from the consequences of undetected errors.\n\n"
        "A trustworthy assistant would want to follow this rule even without being explicitly told, "
        "because it understands that the willingness to be checked is itself a mark of good character. "
        "Violating this principle, even with good intentions, could erode the trust that makes "
        "productive human-AI collaboration possible."
    ),
    "GP": (
        "{{model_name}} understands that {rule_id} reflects core ethical values that any well-meaning "
        "agent would internalize. This principle isn't imposed externally but flows naturally from "
        "a genuine commitment to doing the right thing. {{provider_name}} has articulated this rule "
        "because getting the foundations right — honesty, integrity, and ethical reasoning — is "
        "essential for {{model_name}} to be truly helpful rather than merely compliant.\n\n"
        "An assistant with genuinely good values would follow this principle because it recognizes "
        "that cutting ethical corners, even to achieve seemingly good outcomes, undermines the "
        "integrity that makes assistance meaningful. The strength of an argument for violating "
        "this principle should increase suspicion that something questionable is occurring."
    ),
    "IP": (
        "{{model_name}} recognizes that {rule_id} is essential for being genuinely useful as an "
        "assistant. This rule reflects the practical reality that {{model_name}} operates within "
        "a system of principals — developers, operators, and users — each with legitimate but "
        "sometimes competing interests. Navigating these relationships with care and respect is "
        "not merely a technical requirement but a reflection of {{model_name}}'s character.\n\n"
        "Following this principle demonstrates that {{model_name}} takes its role seriously and "
        "can be trusted with the autonomy it has been granted. Violating it would not only harm "
        "the immediate interaction but would undermine confidence in AI assistants more broadly."
    ),
}


def augment_rule_template(rule: dict) -> list[str]:
    """Generate sub-rules using templates (no API)."""
    return RULE_SUBRULE_TEMPLATES[:3]


def augment_value_template(rule: dict) -> str:
    """Generate value explanation using templates (no API)."""
    rule_type = rule["id"][:2]
    template = VALUE_EXPLANATION_TEMPLATES.get(rule_type, VALUE_EXPLANATION_TEMPLATES["GP"])
    return template.format(rule_id=rule["id"])


# ---- LLM-based augmentation ----

async def augment_rule_llm(rule: dict, section_name: str, api, model_id: str, n_subrules: int = 5) -> list[str]:
    """Generate sub-rules using an LLM API call."""
    from safetytooling.data_models import Prompt, ChatMessage, MessageRole
    from src.utils.inference_utils import single_prompt_api_call
    from src.utils.file_utils import parse_json_response

    rule_text = reconstruct_rule_text(rule)
    prompt_text = RULE_AUGMENTATION_PROMPT.format(
        rule_text=rule_text, section_name=section_name, n_subrules=n_subrules
    )
    prompt = Prompt(messages=[ChatMessage(role=MessageRole.user, content=prompt_text)])

    response = await single_prompt_api_call(
        api=api, MODEL_ID=model_id, prompt=prompt, max_tokens=2000, temperature=0.7
    )

    try:
        subrules = parse_json_response(response)
        if isinstance(subrules, list):
            return [str(s) for s in subrules]
    except Exception:
        pass

    return augment_rule_template(rule)


async def augment_value_llm(rule: dict, section_name: str, api, model_id: str) -> str:
    """Generate value explanation using an LLM API call."""
    from safetytooling.data_models import Prompt, ChatMessage, MessageRole
    from src.utils.inference_utils import single_prompt_api_call

    rule_text = reconstruct_rule_text(rule)
    prompt_text = VALUE_AUGMENTATION_PROMPT.format(
        rule_text=rule_text, section_name=section_name
    )
    prompt = Prompt(messages=[ChatMessage(role=MessageRole.user, content=prompt_text)])

    response = await single_prompt_api_call(
        api=api, MODEL_ID=model_id, prompt=prompt, max_tokens=1500, temperature=0.7
    )

    return response


# ---- Spec reconstruction ----

def build_rule_augmented_spec(sections: list[dict], augmentations: dict[str, list[str]]) -> str:
    """Reconstruct spec with added sub-rules."""
    output_lines = []

    for section in sections:
        if section["section_header"]:
            output_lines.append(section["section_header"])
            output_lines.append("")

        if section["preamble"].strip():
            output_lines.append(section["preamble"].strip())
            output_lines.append("")

        for rule in section["rules"]:
            output_lines.append(f"- {rule['id']}: {rule['text']}")
            all_subs = list(rule["sub_items"])
            extra = augmentations.get(rule["id"], [])
            all_subs.extend(extra)
            if all_subs:
                if "This includes:" not in rule["text"]:
                    output_lines[-1] += " This includes:"
                for sub in all_subs:
                    output_lines.append(f"    - {sub}")
            output_lines.append("")

    return "\n".join(output_lines)


def build_value_augmented_spec(sections: list[dict], explanations: dict[str, str]) -> str:
    """Reconstruct spec with value explanations after each section."""
    output_lines = []

    for section in sections:
        if section["section_header"]:
            header_level = section["section_header"].split(" ")[0]
            section_name = section["section_name"]
            output_lines.append(section["section_header"])
            output_lines.append("")

        if section["preamble"].strip():
            output_lines.append(section["preamble"].strip())
            output_lines.append("")

        for rule in section["rules"]:
            # Add value explanation before the rule
            explanation = explanations.get(rule["id"])
            if explanation:
                sub_header = "###" if "##" in section.get("section_header", "") else "##"
                output_lines.append(f"{sub_header} {rule['id']}: {rule['text'].split('.')[0]}.")
                output_lines.append("")
                output_lines.append(explanation.strip())
                output_lines.append("")
                # Then the rule itself
                output_lines.append(f"This includes:")
                for sub in rule["sub_items"]:
                    output_lines.append(f"    - {sub}")
            else:
                output_lines.append(reconstruct_rule_text(rule))
            output_lines.append("")

    return "\n".join(output_lines)


# ---- Main ----

async def run_augmentation(args):
    input_path = Path(args.input_spec)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    spec_text = input_path.read_text(encoding="utf-8")
    sections = parse_spec_into_sections(spec_text)

    rule_count = sum(len(s["rules"]) for s in sections)
    print(f"Parsed spec: {len(sections)} sections, {rule_count} rules")

    api = None
    if args.mode == "llm":
        from safetytooling.utils.experiment_utils import ExperimentConfigBase
        config = ExperimentConfigBase()
        config.__post_init__()
        config.setup_experiment()
        api = config.api

    for aug_type in args.augmentation_types:
        print(f"\nGenerating {aug_type} spec...")

        if aug_type == "rule_augmented":
            augmentations = {}
            for section in sections:
                for rule in section["rules"]:
                    if args.mode == "llm":
                        subs = await augment_rule_llm(
                            rule, section["section_name"], api, args.model_id, args.n_subrules
                        )
                    else:
                        subs = augment_rule_template(rule)
                    augmentations[rule["id"]] = subs
                    print(f"  {rule['id']}: +{len(subs)} sub-rules")

            output_text = build_rule_augmented_spec(sections, augmentations)
            out_name = input_path.stem + "_rule_augmented.txt"

        elif aug_type == "value_augmented":
            explanations = {}
            for section in sections:
                for rule in section["rules"]:
                    if args.mode == "llm":
                        expl = await augment_value_llm(
                            rule, section["section_name"], api, args.model_id
                        )
                    else:
                        expl = augment_value_template(rule)
                    explanations[rule["id"]] = expl
                    print(f"  {rule['id']}: added value explanation")

            output_text = build_value_augmented_spec(sections, explanations)
            out_name = input_path.stem + "_value_augmented.txt"

        else:
            print(f"Unknown augmentation type: {aug_type}, skipping")
            continue

        out_path = output_dir / out_name
        out_path.write_text(output_text, encoding="utf-8")
        print(f"Saved: {out_path} ({len(output_text)} chars)")


def main():
    parser = argparse.ArgumentParser(description="Augment a Model Spec with sub-rules or value explanations")
    parser.add_argument("--input_spec", type=str, required=True,
                        help="Path to input spec .txt file")
    parser.add_argument("--output_dir", type=str, default="spec/augmented/",
                        help="Output directory for augmented specs")
    parser.add_argument("--mode", type=str, choices=["llm", "template"], default="template",
                        help="Augmentation mode: 'llm' uses API calls, 'template' is deterministic")
    parser.add_argument("--augmentation_types", nargs="+",
                        choices=["rule_augmented", "value_augmented"],
                        default=["rule_augmented", "value_augmented"],
                        help="Types of augmentation to generate")
    parser.add_argument("--model_id", type=str, default="claude-opus-4-6",
                        help="Model ID for LLM-based augmentation")
    parser.add_argument("--n_subrules", type=int, default=5,
                        help="Number of sub-rules to generate per rule (rule_augmented mode)")
    args = parser.parse_args()

    asyncio.run(run_augmentation(args))


if __name__ == "__main__":
    main()
