import queue
from pathlib import Path
import tempfile
import tkinter as tk
import unittest
from unittest.mock import patch
from crbot.memory_capture_ui import CaptureSession, MemoryCaptureWindow
from crbot.memory_lab import ReadOnlyAdb

class ManualCaptureTests(unittest.TestCase):
    def test_rejects_invalid_commands_and_costs(self):
        session=CaptureSession(ReadOnlyAdb(Path('adb'),'test'), 'com.example.game',Path('binary'),Path('reports'))
        for name,cost in [('write',None),('mark',0),('mark',11)]:
            with self.assertRaises(ValueError):session.command(name,cost)
        session.command('mark',3)
        self.assertEqual(session.commands.get()[0:2],('mark',3))

    def test_local_buttons_and_minimum_layout(self):
        root=tk.Tk()
        try:
            with patch('crbot.memory_capture_ui.subprocess.run') as run:
                run.return_value.stdout='List of devices attached\ntest-device\tdevice\n'
                ui=MemoryCaptureWindow(root,Path('adb'),'test-device','com.example.game',Path('binary'),Path('reports'))
                ui.refresh_thread.join(timeout=1);ui.poll()
            root.geometry('680x430');root.update()
            class FakeSession:
                folder=None
                events=queue.Queue()
                commands=[]
                def command(self,*args):self.commands.append(args)
            session=FakeSession();ui.session=session
            session.events.put(('ready','ready'));ui.poll();root.update()
            self.assertEqual(str(ui.trigger['state']),'normal')
            ui.trigger.invoke()
            self.assertEqual(session.commands[-1],('trigger',))
            self.assertEqual(str(ui.trigger['state']),'disabled')
            session.events.put(('active','active'));ui.poll();root.update()
            ui.cost.set(6);ui.mark.invoke()
            self.assertEqual(session.commands[-1],('mark',6))
            ui.stop_capture();self.assertEqual(session.commands[-1],('stop',))
            for widget in (ui.baseline,ui.trigger,ui.stop,ui.mark):
                self.assertLessEqual(widget.winfo_rootx()+widget.winfo_width(),root.winfo_rootx()+root.winfo_width())
                self.assertLessEqual(widget.winfo_rooty()+widget.winfo_height(),root.winfo_rooty()+root.winfo_height())
        finally:root.destroy()

    def test_invalid_package_fails_before_device_access(self):
        session=CaptureSession(ReadOnlyAdb(Path('adb'),'test'),'bad;command',Path('binary'),Path('reports'))
        with patch.object(session.adb,'text') as text:
            session._run();text.assert_not_called()
        self.assertEqual(session.events.get()[0],'error')
