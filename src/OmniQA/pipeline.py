"""Command dispatcher for the complete data construction workflows."""
import argparse
import runpy
import sys

COMMANDS = {
    "run": ("OmniQA.workflow", "Run selected stages with shared input/output configuration"),
    "caption": ("OmniQA.stage1_caption", "Semantic/scene segmentation and fine-grained audiovisual captions"),
    "generate": ("OmniQA.stage2_generate", "ROV: task selection, reasoning graphs, grounding, caption-consistency and question-only gates"),
    "adapt": ("OmniQA.generation.adaptation_image", "RBY: video-derived problems, verification and generated figures"),
    "bridge-adaptation": ("OmniQA.generation.adaptation_to_accepted", "Convert adaptation results to accepted question records"),
    "verify": ("OmniQA.stage3_verify", "Verify timestamped clues against video/audio clips"),
    "refine": ("OmniQA.verification.refine_questions", "Repair or reject questions using audiovisual verification"),
    "gate": ("OmniQA.verification.modality_gate", "Modality shortcut screening: full-caption, visual-only, audio-only and per-step media checks"),
    "audit": ("OmniQA.verification.audit_link_support", "Report-only audit of the presupposition link"),
    "think": ("OmniQA.stage4_thinking", "Text-only thinking synthesis from captions + verified QA + reasoning steps"),
    "export": ("OmniQA.thinking.export_chatml", "Convert recorded model thinking/response rows to training conversations"),
}


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog="omniqa", description=__doc__)
    parser.add_argument("command", choices=COMMANDS, help="; ".join(f"{k}: {v[1]}" for k, v in COMMANDS.items()))
    if not arguments or arguments[0] in ("-h", "--help"):
        parser.print_help()
        return 0
    command = parser.parse_args(arguments[:1]).command
    previous = sys.argv
    try:
        sys.argv = [f"omniqa {command}", *arguments[1:]]
        runpy.run_module(COMMANDS[command][0], run_name="__main__")
    except SystemExit as exc:
        return exc.code or 0
    finally:
        sys.argv = previous
    return 0
