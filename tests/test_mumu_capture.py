import ctypes
import unittest
from threading import RLock
from unittest.mock import Mock
from crbot.mumu_capture import MumuCapture
from crbot.adb import MumuDevice


class MumuCaptureTests(unittest.TestCase):
    def capture(self, failure=False):
        capture = object.__new__(MumuCapture)
        capture.lock = RLock()
        capture.handle = 7
        capture.buffer = capture.size = None
        capture.display_function = Mock(return_value=2)
        capture.library = Mock()
        def read(handle, display, length, width, height, buffer):
            self.assertEqual((handle, display), (7, 2))
            ctypes.cast(width, ctypes.POINTER(ctypes.c_int))[0] = 1
            ctypes.cast(height, ctypes.POINTER(ctypes.c_int))[0] = 2
            if length:
                if failure:
                    return 1
                ctypes.memmove(buffer, bytes([0,0,255,255,255,0,0,255]), 8)
            return 0
        capture.library.nemu_capture_display.side_effect = read
        return capture

    def test_pixels_are_oriented_rgb_and_own_their_storage(self):
        capture = self.capture()
        image = capture.capture()
        self.assertEqual(image.size, (1,2))
        self.assertEqual(list(image.getdata()), [(255,0,0),(0,0,255)])
        ctypes.memset(capture.buffer, 0, 8)
        self.assertEqual(image.getpixel((0,0)), (255,0,0))
        capture.capture()
        self.assertEqual(capture.library.nemu_capture_display.call_count, 3)
        capture.close()
        capture.close()
        capture.library.nemu_disconnect.assert_called_once_with(7)
        with self.assertRaises(OSError):
            capture.capture()

    def test_failed_capture_retries_once_then_reports_failure(self):
        capture = self.capture(failure=True)
        with self.assertRaises(OSError):
            capture.capture()
        self.assertEqual(capture.library.nemu_capture_display.call_count, 4)

    def test_ipc_and_cleanup_failure_still_fall_back_to_adb(self):
        import struct
        device = object.__new__(MumuDevice)
        device._ipc_enabled = True
        device._ipc_capture = Mock()
        device._ipc_capture.capture.side_effect = OSError('capture failed')
        device._ipc_capture.close.side_effect = OSError('disconnect failed')
        device.adb = Mock(return_value=struct.pack('<III',1,1,1)+bytes([255,0,0,255]))
        self.assertEqual(device.screenshot_fast().getpixel((0,0)), (255,0,0))
        self.assertFalse(device._ipc_enabled)
        self.assertIsNone(device._ipc_capture)
        self.assertEqual(device.capture_backend, 'adb_raw')
        self.assertEqual(device.capture_ipc_error, 'capture failed')
        device.screenshot_fast()
        self.assertEqual(device.adb.call_count, 2)
