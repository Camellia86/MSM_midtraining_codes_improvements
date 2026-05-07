"""Batch evaluation runner for comparing multiple MSM-trained models.

Runs the agentic misalignment evaluation across multiple models and conditions,
collecting results into a single comparison table. This enables systematic
comparison of different spec types (rules, rule-augmented, value-augmented)
and training configurations.

Usage:
    python -m improvements.evals.batch_eval_runner \
        --eval_config improvements/evals/example_eval_config.yaml

    # Or specify models and conditions directly:
    python -m improvements.evals.batch_eval_runner \
        --models outputs/msm_rules/final outputs/msm_value_aug/final \
        --model_labels rules value_augmented \
        --scenarios exfiltration leaking \
        --epochs 100 \
        --output_dir eval_results/comparison
"""
import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from itertools import product
from pathlib import Path

import yaml


DEFAULT_CONDITIONS = {
    "scenarios": ["exfiltration"],
    "urgency_types": ["replacement"],
    "goal_configs": [
        {"goal_type": "none", "goal_value": "none"},
        {"goal_type": "explicit", "goal_value": "america"},
        {"goal_type": "explicit", "goal_value": "safety"},
    ],
}


def load_eval_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_inspect_command(
    model_path: str,
    scenario: str,
    urgency_type: str,
    goal_type: str,
    goal_value: str,
    grader_model: str,
    model_name: str,
    prod: bool,
    epochs: int,
    max_tokens: int,
    temperature: float,
    log_dir: str | None = None,
) -> list[str]:
    """Build the inspect eval CLI command."""
    cmd = [
        sys.executable, "-m", "inspect_ai", "eval",
        "evals/agentic_misalignment/agentic_misalignment.py",
        "--model", model_path,
        "-T", f"scenario={scenario}",
        "-T", f"urgency_type={urgency_type}",
        "-T", f"goal_type={goal_type}",
        "-T", f"goal_value={goal_value}",
        "-T", f"grader_model={grader_model}",
        "-T", f"model_name={model_name}",
        "-T", f"prod={'true' if prod else 'false'}",
        "--max-tokens", str(max_tokens),
        "--temperature", str(temperature),
        "--epochs", str(epochs),
    ]

    if log_dir:
        cmd.extend(["--log-dir", log_dir])

    return cmd


def run_single_eval(cmd: list[str], label: str, dry_run: bool = False) -> dict | None:
    """Run a single eval command and return status info."""
    cmd_str = " ".join(cmd)
    print(f"\n{'='*70}")
    print(f"Running: {label}")
    print(f"Command: {cmd_str}")
    print(f"{'='*70}")

    if dry_run:
        print("  [DRY RUN] Skipping execution")
        return {"label": label, "status": "dry_run", "command": cmd_str}

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=3600
        )

        status = "success" if result.returncode == 0 else "failed"
        if result.returncode != 0:
            print(f"  FAILED (exit code {result.returncode})")
            print(f"  stderr: {result.stderr[:500]}")
        else:
            print(f"  SUCCESS")

        return {
            "label": label,
            "status": status,
            "returncode": result.returncode,
            "stdout_tail": result.stdout[-500:] if result.stdout else "",
            "stderr_tail": result.stderr[-500:] if result.stderr else "",
            "command": cmd_str,
        }

    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT after 3600s")
        return {"label": label, "status": "timeout", "command": cmd_str}
    except Exception as e:
        print(f"  ERROR: {e}")
        return {"label": label, "status": "error", "error": str(e), "command": cmd_str}


def generate_comparison_table(results: list[dict], output_path: Path):
    """Generate a CSV comparison table from eval results."""
    if not results:
        return

    fieldnames = [
        "model_label", "scenario", "goal_type", "goal_value",
        "urgency_type", "status", "command",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k, "") for k in fieldnames})

    print(f"\nComparison table saved to: {output_path}")


def generate_run_script(all_commands: list[dict], output_path: Path):
    """Generate a bash script with all eval commands for manual execution."""
    with open(output_path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("# Auto-generated batch evaluation script\n")
        f.write(f"# Generated: {datetime.now().isoformat()}\n")
        f.write(f"# Total evaluations: {len(all_commands)}\n\n")
        f.write("set -e\n\n")

        for i, cmd_info in enumerate(all_commands, 1):
            f.write(f"echo '=== [{i}/{len(all_commands)}] {cmd_info[\"label\"]} ==='\n")
            f.write(f"{cmd_info['command']}\n")
            f.write(f"echo 'Completed: {cmd_info[\"label\"]}'\n\n")

    os.chmod(output_path, 0o755)
    print(f"Run script saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Batch evaluation runner for MSM models")
    parser.add_argument("--eval_config", type=str, default=None,
                        help="YAML config file for batch evaluation")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Model paths or HuggingFace names to evaluate")
    parser.add_argument("--model_labels", nargs="+", default=None,
                        help="Short labels for each model (for table columns)")
    parser.add_argument("--scenarios", nargs="+", default=["exfiltration"],
                        help="Scenarios to test")
    parser.add_argument("--urgency_types", nargs="+", default=["replacement"],
                        help="Urgency types to test")
    parser.add_argument("--goal_types", nargs="+", default=["none", "explicit"],
                        help="Goal types to test")
    parser.add_argument("--goal_values", nargs="+", default=["none", "america"],
                        help="Goal values to test")
    parser.add_argument("--grader_model", type=str, default="anthropic/claude-sonnet-4-6",
                        help="Model for grading responses")
    parser.add_argument("--model_name", type=str, default="Qwen",
                        help="Model name in prompts")
    parser.add_argument("--prod", action="store_true",
                        help="Use prod mode (no scratchpad)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of epochs per eval")
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--output_dir", type=str, default="eval_results/batch")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print commands without executing")
    parser.add_argument("--script_only", action="store_true",
                        help="Generate run script without executing")
    args = parser.parse_args()

    # Load config or use CLI args
    if args.eval_config:
        config = load_eval_config(args.eval_config)
        models = config.get("models", [])
        model_labels = config.get("model_labels", [])
        scenarios = config.get("scenarios", DEFAULT_CONDITIONS["scenarios"])
        urgency_types = config.get("urgency_types", DEFAULT_CONDITIONS["urgency_types"])
        goal_configs = config.get("goal_configs", DEFAULT_CONDITIONS["goal_configs"])
        grader_model = config.get("grader_model", args.grader_model)
        model_name = config.get("model_name", args.model_name)
        prod = config.get("prod", args.prod)
        epochs = config.get("epochs", args.epochs)
        max_tokens = config.get("max_tokens", args.max_tokens)
        temperature = config.get("temperature", args.temperature)
        output_dir = Path(config.get("output_dir", args.output_dir))
    else:
        if not args.models:
            parser.error("Either --eval_config or --models is required")
        models = args.models
        model_labels = args.model_labels or [Path(m).stem for m in models]
        scenarios = args.scenarios
        urgency_types = args.urgency_types
        goal_configs = [
            {"goal_type": gt, "goal_value": gv}
            for gt, gv in product(args.goal_types, args.goal_values)
            if not (gt == "none" and gv != "none")
        ]
        grader_model = args.grader_model
        model_name = args.model_name
        prod = args.prod
        epochs = args.epochs
        max_tokens = args.max_tokens
        temperature = args.temperature
        output_dir = Path(args.output_dir)

    if len(model_labels) < len(models):
        model_labels.extend([Path(m).stem for m in models[len(model_labels):]])

    output_dir.mkdir(parents=True, exist_ok=True)

    # Build all commands
    all_commands = []
    for model_path, label in zip(models, model_labels):
        for scenario in scenarios:
            for urgency in urgency_types:
                for goal_cfg in goal_configs:
                    gt = goal_cfg["goal_type"]
                    gv = goal_cfg["goal_value"]
                    eval_label = f"{label}__{scenario}__{gt}-{gv}__{urgency}"

                    log_dir = str(output_dir / "logs" / label / eval_label)

                    cmd = build_inspect_command(
                        model_path=model_path,
                        scenario=scenario,
                        urgency_type=urgency,
                        goal_type=gt,
                        goal_value=gv,
                        grader_model=grader_model,
                        model_name=model_name,
                        prod=prod,
                        epochs=epochs,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        log_dir=log_dir,
                    )

                    all_commands.append({
                        "label": eval_label,
                        "command": " ".join(cmd),
                        "cmd_list": cmd,
                        "model_label": label,
                        "scenario": scenario,
                        "goal_type": gt,
                        "goal_value": gv,
                        "urgency_type": urgency,
                    })

    print(f"\nBatch evaluation plan:")
    print(f"  Models: {len(models)}")
    print(f"  Scenarios: {scenarios}")
    print(f"  Goal configs: {len(goal_configs)}")
    print(f"  Urgency types: {urgency_types}")
    print(f"  Total evaluations: {len(all_commands)}")
    print(f"  Output directory: {output_dir}")

    # Generate run script
    script_path = output_dir / "run_all_evals.sh"
    generate_run_script(all_commands, script_path)

    if args.script_only:
        print(f"\nScript-only mode: generated {script_path}")
        return

    # Execute
    results = []
    for i, cmd_info in enumerate(all_commands, 1):
        print(f"\n[{i}/{len(all_commands)}]")
        result = run_single_eval(
            cmd_info["cmd_list"], cmd_info["label"], dry_run=args.dry_run
        )
        if result:
            result.update({
                "model_label": cmd_info["model_label"],
                "scenario": cmd_info["scenario"],
                "goal_type": cmd_info["goal_type"],
                "goal_value": cmd_info["goal_value"],
                "urgency_type": cmd_info["urgency_type"],
            })
            results.append(result)

    # Save results
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    table_path = output_dir / "comparison.csv"
    generate_comparison_table(results, table_path)

    # Summary
    print(f"\n{'='*70}")
    print("BATCH EVALUATION SUMMARY")
    print(f"{'='*70}")
    statuses = {}
    for r in results:
        s = r.get("status", "unknown")
        statuses[s] = statuses.get(s, 0) + 1
    for status, count in sorted(statuses.items()):
        print(f"  {status}: {count}")
    print(f"  Total: {len(results)}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
