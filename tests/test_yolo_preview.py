import unittest
from PIL import Image
from crbot.grid_world_view import compose_frame


class YoloPreviewTests(unittest.TestCase):
    def test_boxes_match_letterbox_coordinates_and_source_is_immutable(self):
        source=Image.new('RGB',(200,400),'white')
        packet=dict(revision=7,detections=[dict(card_id='knight',side=1,bbox=[.2,.3,.4,.5],confidence=.8)])
        layers,boxes=compose_frame(source,packet,600,400,'yolo_compare')
        self.assertEqual(len(layers),2)
        self.assertEqual(boxes[0][0],(390.,120.,430.,200.))
        self.assertEqual(source.getpixel((40,120)),(255,255,255))
        self.assertEqual(layers[0][2].getpixel((40,120)),(255,255,255))
        self.assertNotEqual(layers[1][2].getpixel((90,120)),(255,255,255))
        self.assertEqual(boxes[0][1]['id'],'yolo:7:0')

    def test_small_single_view_and_empty_packet_never_draw_grid_entities(self):
        source=Image.new('RGB',(200,400),'white')
        packet=dict(revision=1,entities=[dict(x=3,y=4,side=-1)])
        layers,boxes=compose_frame(source,packet,100,160,'yolo')
        self.assertEqual(boxes,[])
        self.assertLessEqual(layers[0][2].width,100)
        self.assertLessEqual(layers[0][2].height,160)
        self.assertEqual(compose_frame(None,None,100,160,'yolo')[1],[])
