from __future__ import annotations

import json
import queue
import tempfile
import threading
import unittest
from collections import deque
from pathlib import Path

from PIL import Image

from crbot.demonstration import (
    DemonstrationRecorder,
    GetEventTouchParser,
    TouchGesture,
    TouchscreenInfo,
    discover_touchscreen,
)


class TouchParserTests(unittest.TestCase):
    def test_protocol_b_tap_is_parsed(self) -> None:
        parser = GetEventTouchParser()
        lines = [
            "[  123.000001] EV_ABS ABS_MT_TRACKING_ID 0000002a",
            "[  123.000010] EV_ABS ABS_MT_POSITION_X 0000012c",
            "[  123.000020] EV_ABS ABS_MT_POSITION_Y 000006ac",
            "[  123.100000] EV_ABS ABS_MT_POSITION_X 00000130",
            "[  123.100010] EV_ABS ABS_MT_POSITION_Y 000006b0",
            "[  123.100020] EV_ABS ABS_MT_TRACKING_ID ffffffff",
        ]

        gestures = [value for line in lines if (value := parser.feed(line)) is not None]

        self.assertEqual(len(gestures), 1)
        gesture = gestures[0]
        self.assertEqual((gesture.start_x, gesture.start_y), (300, 1708))
        self.assertEqual((gesture.end_x, gesture.end_y), (304, 1712))

    def test_touchscreen_discovery_uses_reported_axis_limits(self) -> None:
        output = """
add device 1: /dev/input/event3
  name:     "Mouse"
add device 2: /dev/input/event4
  name:     "Xiaomi Touchscreen"
    ABS (0003): ABS_MT_POSITION_X : value 0, min 0, max 1080, fuzz 0
                ABS_MT_POSITION_Y : value 0, min 0, max 1920, fuzz 0
"""

        class FakeDevice:
            @staticmethod
            def adb(_arguments: list[str], timeout: int = 0) -> str:
                return output

        info = discover_touchscreen(FakeDevice())  # type: ignore[arg-type]

        self.assertEqual(info.path, "/dev/input/event4")
        self.assertEqual((info.maximum_x, info.maximum_y), (1080, 1920))


class DemonstrationActionTests(unittest.TestCase):
    def test_card_selection_and_deploy_become_one_expert_action(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            recorder = DemonstrationRecorder.__new__(DemonstrationRecorder)
            recorder.touchscreen = TouchscreenInfo(
                "/dev/input/event4", "Touchscreen", 1080, 1920
            )
            recorder.config = {
                "demonstration": {"card_region_top": 0.82},
                "vision": {
                    "card_slot_centers": [
                        [0.3, 0.89],
                        [0.5, 0.89],
                        [0.69, 0.89],
                        [0.875, 0.89],
                    ]
                },
            }
            recorder.state_lock = threading.Lock()
            recorder.file_lock = threading.Lock()
            recorder.snapshots = deque(
                [(1.0, Image.new("RGB", (1080, 1920), "black"))], maxlen=12
            )
            recorder.latest_frame = recorder.snapshots[-1][1]
            recorder.in_battle = True
            recorder.offline_verified = True
            recorder.battle_index = 2
            recorder.pending = None
            recorder.jobs = queue.Queue()
            recorder.raw_touch_count = 0
            recorder.raw_touches_path = Path(directory) / "raw_touches.jsonl"
            recorder.stop_event = threading.Event()

            recorder._handle_gesture(
                TouchGesture(1.0, 1.1, 324, 1709, 324, 1709)
            )
            recorder._handle_gesture(
                TouchGesture(1.2, 1.3, 360, 1152, 360, 1152)
            )

            job = recorder.jobs.get_nowait()
            rows = [
                json.loads(line)
                for line in recorder.raw_touches_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(job.slot_index, 0)
            self.assertAlmostEqual(job.deploy_point[0], 1 / 3, places=3)
            self.assertAlmostEqual(job.deploy_point[1], 0.6, places=3)
            self.assertEqual([row["category"] for row in rows], ["card_select", "card_deploy"])
            self.assertTrue(all(row["offline_verified"] for row in rows))


if __name__ == "__main__":
    unittest.main()
