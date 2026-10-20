"""Chain captioning, selectable authors, verification, refinement, modality gating and synthesis."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def stage_arguments(options):
    result = []
    for name, value in options.items():
        name = name.replace("_", "-")
        if any(part in name.lower() for part in ("api-key", "password", "secret")) or name in ("token", "access-token", "auth-token"):
            raise ValueError("Pass credentials through environment variables, not workflow configuration")
        if value is None:
            continue
        if value is False:
            result.append("--no-" + name)
            continue
        result.append("--" + name)
        if value is not True:
            result.extend(str(v) for v in (value if isinstance(value, list) else [value]))
    return result


def build_plan(config):
    stages = set(config.get("stages", [2, 3, 4]))
    if not stages <= {1, 2, 3, 4}:
        raise ValueError("Stages must be chosen from 1, 2, 3, 4")
    mode = config.get("mode", "ROV").upper()
    if mode not in ("ROV", "RBY"):
        raise ValueError("Generation mode must be ROV or RBY")
    run = Path(config["run_dir"])
    captions = Path(config.get("caption_root", "data/caption"))
    dataset = config.get("dataset", "videos")
    generation = run / "generation"
    verified = run / "verification"
    refined = run / "refinement"
    gated = run / "gate"
    selected = Path(config.get("input", generation / "accepted.jsonl"))
    plan = []
    def add(command, defaults):
        overrides = {key.replace("_", "-"): value for key, value in config.get(command, {}).items()}
        chained = {"input", "output", "output-dir", "outdir", "caption-root", "caption-dir", "caption-raw-dir"}
        if any(key in chained and key in defaults and str(value) != str(defaults[key]) for key, value in overrides.items()):
            raise ValueError("Set chained paths through run_dir, caption_root and input; use individual commands for custom paths")
        options = {**defaults, **overrides}
        plan.append([command, *stage_arguments(options)])
    if 1 in stages:
        if not config.get("data_path"):
            raise ValueError("data_path is required for caption generation")
        add("caption", {"data-path": config["data_path"], "output-dir": str(captions / dataset)})
    if 2 in stages:
        command = {"ROV": "generate", "RBY": "adapt"}[mode]
        if mode == "RBY":
            add(command, {"caption-dir": str(captions / "*" / "raw"), "outdir": str(generation)})
            add("bridge-adaptation", {"outdir": str(generation), "output": str(selected)})
        elif mode == "ROV" and os.environ.get("OPENAI_API_BASE"):
            add(command, {"caption-raw-dir": str(captions / "*" / "raw"), "output-dir": str(generation), "openai-api-base": os.environ["OPENAI_API_BASE"]})
        else:
            add(command, {"caption-raw-dir": str(captions / "*" / "raw"), "output-dir": str(generation)})
    if 3 in stages:
        if config.get("enable_verify", True):
            add("verify", {"input": str(selected), "output-dir": str(verified)})
            selected = verified / "accepted_verified.jsonl"
        if config.get("enable_refine", True):
            add("refine", {"input": str(selected), "output-dir": str(refined)})
            selected = refined / "refined_accepted.jsonl"
        if config.get("enable_gate", True):
            add("gate", {"input": str(selected), "output-dir": str(gated), "caption-raw-dir": str(captions / dataset / "raw")})
            selected = gated / "accepted_gated.jsonl"
        if config.get("enable_audit", False):
            # Report-only diagnostic; it does not filter the chained pool.
            add("audit", {"input": str(selected), "output-dir": str(run / "audit")})
    elif 4 in stages and "input" not in config:
        pools = [gated / "accepted_gated.jsonl", refined / "refined_accepted.jsonl", verified / "accepted_verified.jsonl", selected]
        if not config.get("enable_gate", True):
            pools.pop(0)
        selected = next((p for p in pools if p.is_file()), selected)
    if 4 in stages:
        add("think", {"input": str(selected), "output-dir": str(run / "thinking"), "caption-root": str(captions)})
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        plan = build_plan(json.loads(args.config.read_text()))
    except (ValueError, KeyError) as exc:
        parser.error(str(exc))
    for command in plan:
        print(json.dumps(command, ensure_ascii=False), flush=True)
        if not args.dry_run:
            subprocess.run([sys.executable, "-m", "OmniQA", *command], check=True)


if __name__ == "__main__":
    main()
