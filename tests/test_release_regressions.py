"""Offline regressions for prediction safety, scoring, and SFT quality."""
import argparse
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from OmniQA import stage4_thinking as thinking
from OmniReasoningBench.scoring import judge, video_mcq


class VideoMCQTests(unittest.TestCase):
    def test_scores_current_and_legacy_rows_without_changing_input(self):
        rows = [
            {"model_answer": "A", "correct_answer": "A"},
            {"gen": "B", "answer": "B"},
            {"gen": ["A"], "answer": "A"},
            {"raw_output": "B", "answer": "B"},
            {"model_answer": "", "correct_answer": "A"},
            {"model_answer": "A", "correct_answer": "A", "error": "service error"},
        ]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "predictions.jsonl"
            original = "".join(json.dumps({**row, "options": ["first", "second"]}) + "\n"
                               for row in rows).encode()
            path.write_bytes(original)
            with contextlib.redirect_stdout(io.StringIO()):
                result = video_mcq.score_file(path)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(result["omnivideobench/acc"], 4 / 6)
            self.assertEqual(result["omnivideobench/failed"], 2)
            self.assertTrue((path.parent / "predictions_eval_result.json").is_file())

    def test_legacy_output_name_is_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "omnivideobench_output_turn.jsonl"
            path.write_text('{"gen":"A","answer":"A"}\n')
            with contextlib.redirect_stdout(io.StringIO()):
                video_mcq.score_file(path)
            self.assertTrue((path.parent / "omnivideobench_output_eval_result.json").is_file())
            self.assertEqual(json.loads(path.read_text())["gen"], "A")

    def test_output_alias_cannot_overwrite_predictions(self):
        for link in (os.symlink, os.link):
            with self.subTest(link=link.__name__), tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / "predictions.jsonl"
                original = '{"gen":"A","answer":"A"}\n'
                path.write_text(original)
                link(path, path.with_name("predictions_eval_result.json"))
                with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
                    video_mcq.score_file(path)
                self.assertEqual(path.read_text(), original)

    def test_video_parser_ignores_tags_inside_thinking(self):
        for raw in ("<think><answer>A</answer></think>\nB",
                    "<think>\\boxed{A}</think>\nB"):
            with self.subTest(raw=raw):
                self.assertEqual(video_mcq.extract_mcq_answer(raw), "B")
        self.assertIsNone(video_mcq.extract_mcq_answer("<think><answer>A</answer>"))


class JudgeTests(unittest.TestCase):
    def test_direct_scoring_uses_committed_answer(self):
        raw = "<think>I reject A) because it is unsupported.</think>\nB"
        for gold, correct in (("A", False), ("B", True)):
            with self.subTest(gold=gold):
                item = {"answer_format": "mcq", "options": ["A) first", "B) second"],
                        "answer_letters": [gold]}
                result = judge.score_item(item, raw, judge_model=judge.JUDGE_MODEL,
                                          timeout=1, max_retries=1)
                self.assertEqual(result["pred"], "B")
                self.assertEqual(result["correct"], correct)
        self.assertEqual(judge.extract_pred_letters(raw, {"A", "B"}), ["B"])

    def test_plain_and_multiple_answers_keep_working(self):
        self.assertEqual(judge.extract_pred_letters("Final Answer: B", {"A", "B"}), ["B"])
        self.assertEqual(judge.extract_pred_letters("Final Answer: B and C", {"A", "B", "C"}),
                         ["B", "C"])
        self.assertEqual(judge.committed_answer("<think>A</think><think>B"), "")

    def test_open_ended_judge_receives_only_the_answer(self):
        item = {"answer_format": "open_numeric", "question": "How many?", "answer": "10"}
        with patch.object(judge, "llm_judge", return_value=(True, "CORRECT")) as call:
            result = judge.score_item(item, "<think>Final Answer: 5</think>\n10",
                                      judge_model=judge.JUDGE_MODEL, timeout=1, max_retries=1)
        self.assertTrue(result["correct"])
        self.assertEqual(call.call_args.args[2], "10")

    def test_default_cli_and_compatibility_flag(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            items = [{"id": str(i), "answer_format": "mcq", "task_type": "test",
                      "options": ["A) first", "B) second"], "answer_letters": [gold]}
                     for i, gold in enumerate(("A", "B", "A"))]
            raw = "<think>I reject A).</think>\nB"
            preds = [{"id": str(i), "gen": [raw if i < 2 else "<think>A) is possible"]}
                     for i in range(3)]
            refs, predictions = root / "refs.jsonl", root / "raw.jsonl"
            refs.write_text("".join(json.dumps(row) + "\n" for row in items))
            predictions.write_text("".join(json.dumps(row) + "\n" for row in preds))
            for flag in ([], ["--answer_after_think"]):
                output = root / ("compatibility" if flag else "default")
                run = subprocess.run(
                    [sys.executable, "-B", "-m", "OmniReasoningBench", "score-judge",
                     "--references", str(refs), "--predictions", str(predictions),
                     "--out_dir", str(output), "--judge_api_base", "https://example.invalid/v1",
                     *flag], capture_output=True, text=True, timeout=30,
                    env={**os.environ, "JUDGE_MODEL": judge.JUDGE_MODEL})
                self.assertEqual(run.returncode, 0, run.stderr)
                summary = json.loads((output / "summary.json").read_text())
                self.assertEqual(summary["overall"]["n"], 3)
                self.assertEqual(summary["overall"]["correct"], 1)
                self.assertEqual(summary["no_answer_truncated"], 1)


class ThinkingTests(unittest.TestCase):
    @staticmethod
    def trace(letter):
        return (f"Got it!\n{thinking.WATCH}\nI see a person.\n"
                f"{thinking.ANALYZE}\nThis points to option {letter}.")

    def synthesize(self, responses):
        row = {"video_id": "sample", "source": "videos", "video_path": "sample.mp4",
               "candidate": {"question": "Which option?", "options": ["first", "second"],
                             "correct_option": "A"}}
        args = argparse.Namespace(caption_root=Path("unused"), gen_tries=len(responses))
        with patch.object(thinking, "load_caption_bundle", return_value={}), \
                patch.object(thinking, "call_llm", side_effect=responses) as call:
            result = thinking.synth_one(row, 0, args, thinking.RateLimiter(0))
        return result, call.call_count

    def test_wrong_conclusion_triggers_regeneration(self):
        result, calls = self.synthesize([self.trace("B"), self.trace("A")])
        self.assertEqual(calls, 2)
        self.assertTrue(thinking.is_clean_sft(result))
        self.assertTrue(result["concluded_matches_gold"])

    def test_wrong_missing_and_last_conclusions_are_rejected(self):
        traces = [self.trace("B"), self.trace("A").replace("This points to option A.", "Done."),
                  self.trace("A") + "\nThis points to option B.",
                  self.trace("A") + "\nActually, I choose B."]
        for trace in traces:
            with self.subTest(trace=trace):
                result, _ = self.synthesize([trace])
                self.assertFalse(thinking.is_clean_sft(result))
                self.assertTrue(result["qc_issues"])
        result, _ = self.synthesize([self.trace("a")])
        self.assertTrue(thinking.is_clean_sft(result))

    def test_old_inconsistent_cache_is_excluded_from_sft_and_resume(self):
        good, _ = self.synthesize([self.trace("A")])
        bad = {**good, "idx": 1, "think": self.trace("B"), "concluded": "A",
               "concluded_matches_gold": True, "qc_issues": []}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            input_path = root / "input.jsonl"
            input_path.write_text("")
            cache = root / "synth_all.jsonl"
            cache.write_text("".join(json.dumps(row) + "\n" for row in (good, bad)))
            self.assertEqual(thinking.load_done_keys(cache), {"videos::sample::0"})
            with patch.object(sys, "argv", ["think", "--input", str(input_path),
                                           "--output-dir", str(root), "--api-key", "offline"]), \
                    patch.object(thinking, "call_llm", side_effect=AssertionError("Unexpected API call")), \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(thinking.main(), 0)
            rows = (root / "sft.jsonl").read_text().splitlines()
            self.assertEqual(len(rows), 1)
            self.assertEqual(json.loads(rows[0])["messages"], good["sft_messages"])
            summary = json.loads((root / "summary.json").read_text())
            self.assertEqual(summary["qc_fail"], 1)


if __name__ == "__main__":
    unittest.main()
