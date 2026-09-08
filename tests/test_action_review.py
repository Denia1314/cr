from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from crbot.action_review import audit_action_review, export_action_review


class ActionReviewTests(unittest.TestCase):
    def test_export_is_stratified_and_never_overwrites(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "runs" / "r1"
            run.mkdir(parents=True)
            with (run / "replay_transitions.jsonl").open("w", encoding="utf-8") as handle:
                for index, status in enumerate(("confirmed", "unknown", "rejected", "confirmed")):
                    handle.write(json.dumps({
                        "transition_id": f"t{index}",
                        "battle_index": index + 1,
                        "action_status": status,
                        "action": {"action_id": f"a{index}", "action_status": status},
                    }) + "\n")
            with (run / "events.jsonl").open("w", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "event": "battle_action_proposed", "action_id": "a0", "frame": "frames/pre.jpg"
                }) + "\n")
                handle.write(json.dumps({
                    "event": "battle_action_sent", "action_id": "a0", "frame": "frames/post.jpg",
                    "confirmation_frame_role": "final_observation",
                }) + "\n")
            output = root / "review.jsonl"

            result = export_action_review(root, output, limit=4)

            self.assertEqual(result["sampled_attempts"], 4)
            self.assertEqual(set(result["status_counts"]), {"confirmed", "unknown", "rejected"})
            exported = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            linked = next(row for row in exported if row["action_id"] == "a0")
            self.assertTrue(linked["pre_action_frame"].endswith("pre.jpg"))
            self.assertTrue(linked["post_confirmation_frame"].endswith("post.jpg"))
            with self.assertRaisesRegex(ValueError, "未覆盖"):
                export_action_review(root, output, limit=4)

    def test_audit_reports_precision_recall_unknown_and_battle_interval(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "review.jsonl"
            rows = [
                ("b1", "confirmed", True),
                ("b1", "confirmed", False),
                ("b2", "unknown", True),
                ("b2", "rejected", False),
                ("b3", "unknown", "unknown"),
            ]
            with path.open("w", encoding="utf-8") as handle:
                for index, (battle, status, actual) in enumerate(rows):
                    handle.write(json.dumps({
                        "transition_id": f"t{index}",
                        "run": "r",
                        "battle_index": battle,
                        "action_status": status,
                        "review_actual_success": actual,
                    }) + "\n")

            result = audit_action_review(path)

            self.assertEqual(result["labeled_attempts"], 4)
            self.assertEqual(result["review_unknown_attempts"], 1)
            self.assertEqual(result["confirmation_precision"], 0.5)
            self.assertEqual(result["successful_action_recall"], 0.5)
            self.assertEqual(result["system_unknown_rate"], 0.4)
            self.assertIsNotNone(result["battle_cluster_bootstrap_95"]["precision"])


if __name__ == "__main__":
    unittest.main()
