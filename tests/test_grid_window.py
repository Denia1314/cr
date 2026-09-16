import tkinter as tk
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
from PIL import Image
from crbot.grid_world_view import GridWorldWindow


class GridWindowTests(unittest.TestCase):
    def test_embedded_view_fits_small_canvas_and_clears_stale_packet(self):
        root=tk.Tk();self.addCleanup(root.destroy)
        root.geometry('540x160')
        image=Image.new('RGB',(100,200),'#123456')
        packet=dict(revision=1,entities=[dict(id='track:1',x=3,y=4,side=-1,hp_fraction=.8)])
        holder=[(image,packet)]
        window=GridWorldWindow(root,lambda:holder[0],embedded=True)
        window.window.pack(fill='both',expand=True)
        root.update_idletasks();root.update();window.refresh()
        self.assertIsInstance(window.window,tk.Frame)
        self.assertEqual(window.photo.width(),window.canvas.winfo_width())
        self.assertEqual(window.photo.height(),window.canvas.winfo_height())
        self.assertTrue(window.boxes)
        b=window.boxes[0][0]
        window.select(SimpleNamespace(x=(b[0]+b[2])/2,y=(b[1]+b[3])/2))
        root.update_idletasks()
        self.assertTrue(window.details.winfo_ismapped())
        window.hide_details()
        self.assertFalse(window.details.winfo_ismapped())
        holder[0]=(image,None);window.refresh()
        self.assertIsNone(window.packet)
        self.assertEqual(window.boxes,[])
        self.assertIn('等待',window.label.cget('text'))

    def test_deadline_subtracts_render_time_and_recovers_one_late_frame(self):
        for now, expected in ((.02,14),(.04,1),(.2,1)):
            window=SimpleNamespace(window=Mock(),refresh=Mock(),next_tick=0.,period=1/30,
                                   meter_started=0.,updated_frames=0,tick=Mock())
            with patch('crbot.grid_world_view.time.perf_counter',return_value=now):
                GridWorldWindow.tick(window)
            self.assertEqual(window.window.after.call_args.args[0],expected)

    def test_details_follow_identity_and_canvas_does_not_accumulate(self):
        root=tk.Tk();root.withdraw()
        self.addCleanup(root.destroy)
        image=Image.new('RGB',(100,200))
        packet=dict(revision=1,entities=[dict(id='track:1',x=3,y=4,side=-1,hp_fraction=.8)])
        holder=[(image,packet)]
        window=GridWorldWindow(root,lambda:holder[0])
        window.window.geometry('800x600')
        root.update_idletasks()
        window.refresh()
        window.selected_id='track:1'
        holder[0]=(image,dict(revision=2,entities=[dict(packet['entities'][0],hp_fraction=.4)]))
        window.refresh()
        self.assertIn('0.4',window.details.get('1.0','end'))
        self.assertEqual(len(window.canvas.find_all()),1)
        self.assertLessEqual(window.label.winfo_reqwidth(),800)
        holder[0]=(image,dict(revision=3,entities=[]))
        window.refresh()
        self.assertIn('已不在',window.details.get('1.0','end'))
        window.window.destroy()
        self.assertIsNone(window.timer_api)

    def test_stalled_source_reports_zero_new_frames(self):
        window=SimpleNamespace(window=Mock(),refresh=Mock(),next_tick=0.,period=1/30,
                               meter_started=0.,updated_frames=0,tick=Mock(),update_label=Mock())
        with patch('crbot.grid_world_view.time.perf_counter',return_value=2.):
            GridWorldWindow.tick(window)
        self.assertEqual(window.source_fps,0)
