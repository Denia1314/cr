import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
from PIL import Image
from crbot.real_battle_dataset import label_unit, convert_shapes, export_real
from crbot.slice_dataset import REVISION


class RealBattleDatasetTests(unittest.TestCase):
    def test_whole_battles_remain_separate_and_dimensions_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, slices = root/'source', root/'slices'
            slices.mkdir()
            (slices/'manifest.json').write_text(json.dumps({'classes':['knight']}))
            source.mkdir()
            (source/'LICENSE').write_text('MIT fixture')
            for group in range(5):
                folder = source/'images/part2'/f'video/{group}'
                folder.mkdir(parents=True)
                for frame in range(3):
                    Image.new('RGB',(100,200),(group*40,frame*40,0)).save(folder/f'{frame}.jpg')
                    (folder/f'{frame}.json').write_text(json.dumps(dict(imageWidth=100,imageHeight=200,
                        shapes=[dict(label='knight0',shape_type='rectangle',points=[[10,20],[30,40]])])))
            with patch('subprocess.check_output',return_value=REVISION):
                result = export_real(source,slices,root/'export',per_battle=2)
            train={r['group'] for r in result['records'] if r['split']=='train'}
            val={r['group'] for r in result['records'] if r['split']=='val'}
            self.assertFalse(train & val)
            self.assertEqual(result['counts'],{'train':8,'val':2})
            self.assertEqual(len(result['records']),10)
            with patch('subprocess.check_output',return_value=REVISION):
                focused = export_real(source,slices,root/'focused',per_battle=1,focus_units=['knight'])
            self.assertEqual(focused['counts'], {'train':12,'val':3})
            self.assertEqual({r['group'] for r in focused['records'] if r['split']=='train'}, train)
            self.assertEqual({r['group'] for r in focused['records'] if r['split']=='val'}, val)

    def test_states_and_team_do_not_change_identity_or_evolution(self):
        names = ['knight', 'knight-evolution', 'ice-spirit']
        self.assertEqual(label_unit('knight-evolution1_attack', names), 'knight-evolution')
        self.assertEqual(label_unit('ice-spirit0_freeze', names), 'ice-spirit')
        self.assertIsNone(label_unit('knight-unknown0', names))
        self.assertIsNone(label_unit('text0', names))

    def test_rectangle_conversion_clamps_reversed_coordinates(self):
        payload = dict(imageWidth=100, imageHeight=200, shapes=[
            dict(label='knight1_attack', shape_type='rectangle', points=[[110,180],[-10,20]]),
            dict(label='text0', shape_type='rectangle', points=[[0,0],[10,10]])])
        rows, ignored = convert_shapes(payload, ['knight'])
        self.assertEqual(ignored, 1)
        values = rows[0].split()
        self.assertEqual(values[0], '0')
        self.assertEqual([float(v) for v in values[1:]], [.5,.5,1.,.8])
        payload['shapes'][0]['points'] = [[10,10],[10,20]]
        with self.assertRaisesRegex(ValueError, 'Empty'):
            convert_shapes(payload, ['knight'])
