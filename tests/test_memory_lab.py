from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from crbot.memory_lab import (
    FixtureReceiver, LabError, MAX_PACKET, diagnose, package_metadata,
    parse_start_ticks, run_arguments, self_test,
)


def sample(seq=1, **overrides):
    value = {"schema": "memory_lab.fixture.v1", "source": "private_child_fixture", "session": "123-456",
             "sequence": seq, "read_begin_ns": seq * 10000, "read_end_ns": seq * 10000 + 1000,
             "complete": True, "consistent": True, "elixir_milli": seq * 137 % 10001,
             "hand": [(seq + slot) % 8 + 1 for slot in range(4)]}
    value.update(overrides)
    return json.dumps(value).encode()


def summary(count=2, **overrides):
    value = {"schema": "memory_lab.summary.v1", "samples": count, "passed": True,
             "bad_address_rejected": True, "partial_read_rejected": True,
             "exited_child_rejected": True, "game_memory_read": False}
    value.update(overrides)
    return json.dumps(value).encode()


class ReceiverTests(unittest.TestCase):
    def test_complete_fixture_does_not_claim_game_or_end_to_end_age(self):
        receiver = FixtureReceiver(2)
        receiver.accept(sample(), 100.)
        receiver.accept(sample(2), 100.1)
        receiver.accept(summary(), 100.2)
        report = receiver.finish(0)
        self.assertEqual(report["samples"], 2)
        self.assertFalse(report["game_memory_read"])
        self.assertFalse(report["game_fields_verified"])
        self.assertFalse(report["end_to_end_age_verified"])
        self.assertAlmostEqual(report["host_delivery_span_s"], .1)

    def test_bad_packets_are_rejected(self):
        changes = [dict(source="game"), dict(schema="unknown"), dict(complete=False),
                   dict(consistent=False), dict(sequence=True), dict(elixir_milli=1),
                   dict(hand=[True, 3, 4, 5]), dict(hand=[2, 3]), dict(read_end_ns=1),
                   dict(read_begin_ns=-1), dict(session="invalid"), dict(sequence=3)]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(LabError):
                FixtureReceiver(2).accept(sample(**change), 1.)

    def test_duplicate_out_of_order_and_changed_session(self):
        for value in (sample(), sample(3), sample(2, session="123-457"), sample(2, read_begin_ns=1)):
            receiver = FixtureReceiver(3)
            receiver.accept(sample(), 1.)
            with self.assertRaises(LabError):
                receiver.accept(value, 2.)

    def test_missing_summary_missing_samples_and_exit_error(self):
        with self.assertRaises(LabError):
            FixtureReceiver(1).finish(0)
        receiver = FixtureReceiver(2)
        receiver.accept(sample(), 1.)
        with self.assertRaises(LabError):
            receiver.accept(summary(), 2.)
        receiver = FixtureReceiver(1)
        receiver.accept(sample(), 1.)
        receiver.accept(summary(1), 2.)
        with self.assertRaises(LabError):
            receiver.finish(3)
        with self.assertRaises(LabError):
            receiver.accept(sample(2), 3.)

    def test_negative_checks_must_pass(self):
        for key in ("passed", "bad_address_rejected", "partial_read_rejected", "exited_child_rejected"):
            receiver = FixtureReceiver(1)
            receiver.accept(sample(), 1.)
            with self.assertRaises(LabError):
                receiver.accept(summary(1, **{key: False}), 2.)

    def test_malformed_oversized_and_nonfinite_packets(self):
        for value in (b"{", b"\xff", b"[]", b"null", b"x" * (MAX_PACKET + 1), b'{"sequence":NaN}'):
            with self.subTest(value=value[:40]), self.assertRaises(LabError):
                FixtureReceiver(1).accept(value, 1.)


class FakeAdb:
    def __init__(self, *, maps=False, memory=False, changed=False):
        self.serial = "test-device"
        self.maps, self.memory, self.changed = maps, memory, changed
        self.stat_queries = 0
        self.calls = []

    def run(self, *args, **kwargs):
        self.calls.append(args)
        output, code = "", 0
        if args[:2] == ("shell", "pidof"):
            output = "123\n"
        elif args[:2] == ("shell", "cat"):
            self.stat_queries += 1
            tick = 457 if self.changed and self.stat_queries > 1 else 456
            output = f"123 (fixture worker) S " + "0 " * 18 + str(tick)
        elif args[:3] == ("shell", "sh", "-c"):
            code = 0 if (self.maps if "maps" in args[-1] else self.memory) else 1
        return subprocess.CompletedProcess(args, code, output, "")

    def text(self, *args):
        if args[:3] == ("shell", "pm", "list"):
            return "package:com.supercell.clashroyale"
        if args[:2] == ("shell", "dumpsys"):
            return "versionName=1.2 versionCode=123 primaryCpuAbi=arm64-v8a flags=[ HAS_CODE ]"
        if args == ("shell", "id"):
            return "uid=2000(shell)"
        if args == ("shell", "getenforce"):
            return "Enforcing"
        return "x86_64"


class DiagnosisTests(unittest.TestCase):
    def native_report(self, sample_header=True, **changes):
        class NativeAdb(FakeAdb):
            def run(self, *args, **kwargs):
                if args[0] == "exec-out":
                    access = {"schema": "memory_lab.probe.v1", "identity_matches": True,
                              "maps_opened": True, "memory_file_opened": True, "actual_memory_read": True,
                              "platform_header_verified": True, "sample_bytes_read": 16}
                    access.update(changes)
                    return subprocess.CompletedProcess(args, 0, json.dumps(access), "")
                return super().run(*args, **kwargs)
        with patch("crbot.memory_lab.deploy_fixture", return_value="/test-fixture"):
            return diagnose(NativeAdb(), None, Path("fixture"), sample_header)

    def test_explicit_platform_header_does_not_claim_game_fields(self):
        report = self.native_report()
        self.assertEqual(report["status"], "platform_header_verified")
        self.assertTrue(report["game_memory_read"])
        self.assertFalse(report["game_fields_verified"])

    def test_default_probe_rejects_unrequested_memory_bytes(self):
        with self.assertRaises(LabError):
            self.native_report(sample_header=False)

    def test_partial_header_or_changed_identity_is_not_verified(self):
        self.assertEqual(self.native_report(sample_bytes_read=8)["status"], "platform_header_unverified")
        self.assertEqual(self.native_report(identity_matches=False)["status"], "process_changed")

    def test_platform_header_requires_explicit_collector(self):
        with self.assertRaises(LabError):
            diagnose(FakeAdb(), None, None, True)

    def test_stat_with_parentheses_and_spaces(self):
        text = "123 (has ) spaces ( ) S " + "0 " * 18 + "456 7 8"
        self.assertEqual(parse_start_ticks(text), 456)
        for value in ("", "123 no-comm", "123 (a) S 0", "123 (a) S " + "0 " * 19):
            with self.assertRaises(LabError):
                parse_start_ticks(value)

    def test_package_metadata_ignores_permission_flags(self):
        self.assertFalse(package_metadata("flags=[ HAS_CODE ] permission: flags=[ DEBUGGABLE ]")["debuggable"])
        self.assertTrue(package_metadata("pkgFlags=[ HAS_CODE DEBUGGABLE ]")["debuggable"])
        self.assertIsNone(package_metadata("")["debuggable"])

    def test_maps_visible_does_not_claim_memory_readable(self):
        report = diagnose(FakeAdb(maps=True), "com.supercell.clashroyale", None)
        self.assertEqual(report["status"], "access_blocked")
        self.assertFalse(report["game_memory_read"])
        self.assertFalse(report["game_fields_verified"])

    def test_open_only_still_does_not_claim_actual_read(self):
        adb = FakeAdb(maps=True, memory=True)
        report = diagnose(adb, None, None)
        self.assertEqual(report["status"], "access_open_only")
        self.assertFalse(report["access"]["actual_memory_read"])
        self.assertFalse(report["settings_changed"])
        self.assertFalse(any("root" in call or "connect" in call or "push" in call for call in adb.calls))

    def test_pid_reuse_invalidates_access(self):
        report = diagnose(FakeAdb(maps=True, memory=True, changed=True), None, None)
        self.assertEqual(report["status"], "process_changed")

    def test_shell_injection_package_is_rejected_before_queries(self):
        adb = FakeAdb()
        with self.assertRaises(LabError):
            diagnose(adb, "com.example;id", None)
        self.assertEqual(adb.calls, [])

    def test_report_cannot_overwrite_code_or_config(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = argparse.Namespace(config=str(root / "config.json"), output=root / "config.json")
            with self.assertRaises(LabError):
                run_arguments(args)
            self.assertFalse((root / "config.json").exists())

    def test_self_test_rejects_unbounded_duration_before_adb(self):
        adb = FakeAdb()
        for count, gap in ((0, 100), (601, 100), (600, 1000), (50, 0)):
            with self.assertRaises(LabError):
                self_test(adb, Path("unused"), count, gap)
        self.assertEqual(adb.calls, [])


if __name__ == "__main__":
    unittest.main()
