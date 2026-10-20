"""OmniReasoningBench command-line entry point."""

import runpy
import sys


COMMANDS = {
    "infer": ("OmniReasoningBench.backends.sglang",
              "run over_video / beyond_video (or OmniVideoBench / WorldSense) against an SGLang/OpenAI endpoint"),
    "score-judge": ("OmniReasoningBench.scoring.judge",
                    "rule-matched MCQ; deepseek-v4-flash for open-ended answers, numeric fallback on judge failure"),
    "score-video-mcq": ("OmniReasoningBench.scoring.video_mcq",
                        "OmniVideoBench / WorldSense type and duration reports"),
    "report-over-video": ("OmniReasoningBench.reporting.over_video",
                          "reasoning over video: breakdown by answer_format / task_type / source"),
    "report-beyond-video": ("OmniReasoningBench.reporting.beyond_video",
                            "reasoning beyond video: breakdown by answer_format / figure / task_type"),
    "report-table": ("OmniReasoningBench.reporting.table",
                     "pool both settings into the MCQ / OE / Overall table row"),
}


def usage():
    lines = ["usage: omnireasoning COMMAND [ARGS]", "", __doc__, "", "commands:"]
    lines += [f"  {name:<20} {help_text}" for name, (_, help_text) in COMMANDS.items()]
    lines += ["", "Use `omnireasoning COMMAND --help` for the arguments of each command."]
    return "\n".join(lines)


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] in ("-h", "--help"):
        print(usage())
        return 0 if arguments else 2
    if arguments[0] not in COMMANDS:
        print(usage(), file=sys.stderr)
        print(f"\nomnireasoning: unknown command {arguments[0]!r}", file=sys.stderr)
        return 2
    previous = sys.argv
    try:
        sys.argv = ["omnireasoning " + arguments[0], *arguments[1:]]
        runpy.run_module(COMMANDS[arguments[0]][0], run_name="__main__")
    except SystemExit as exc:
        return exc.code or 0
    finally:
        sys.argv = previous
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
