import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from crbot.rare_unit_dataset import enrich_real
from crbot.slice_dataset import REVISION


class RareUnitDatasetTests(unittest.TestCase):
    def test_only_training_battles_added_validation_unchanged_duplicate_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory);base=root/'base';source=root/'source';out=root/'out'
            for split in ['train', 'val']:
                (base/'images'/split).mkdir(parents=True)
                (base/'labels'/split).mkdir(parents=True)
                Image.new('RGB', (20, 20), 'red' if split=='train' else 'blue').save(base/'images'/split/'original.jpg')
                (base/'labels'/split/'original.txt').write_text('0 .5 .5 .5 .5\n')
            records=[dict(group=group, split=split, source=f'{group}/original.json',
                          sha256=hashlib.sha256((base/'images'/split/'original.jpg').read_bytes()).hexdigest())
                     for group,split in [('training', 'train'), ('holdout', 'val')]]
            (base/'manifest.json').write_text(json.dumps(dict(kind='real_battle_bootstrap',revision=REVISION,
                classes=['knight','mega-knight'],counts=dict(train=1,val=1),records=records)))
            for group in ['training','holdout']:
                folder=source/'images/part2'/group;folder.mkdir(parents=True)
                for frame in [30,31,80]:
                    Image.new('RGB',(20,20),(frame,40,80)).save(folder/f'{frame}.jpg')
                    (folder/f'{frame}.json').write_text(json.dumps(dict(imageWidth=20,imageHeight=20,
                        shapes=[dict(label='mega-knight1',shape_type='rectangle',points=[[2,2],[12,12]])])))
            # A training frame identical to a held-out image must never enter training.
            folder=source/'images/part2/training'
            (folder/'80.jpg').write_bytes((base/'images/val/original.jpg').read_bytes())
            with patch('subprocess.check_output',return_value=REVISION):
                result=enrich_real(base,source,out,target_boxes=10,max_extra=10)
            self.assertEqual(result['added_images'],1)
            self.assertEqual((out/'images/val/original.jpg').read_bytes(),(base/'images/val/original.jpg').read_bytes())
            self.assertEqual((out/'labels/val/original.txt').read_bytes(),(base/'labels/val/original.txt').read_bytes())
            m=json.loads((out/'manifest.json').read_text())
            self.assertTrue(all(r['group']=='training' for r in m['records'][2:]))
            self.assertEqual(result['classes'][1]['after'],1)
            self.assertEqual(len(list((base/'images/train').glob('*.jpg'))),1)

    def test_cross_split_groups_rejected_without_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);base=root/'base';base.mkdir()
            (base/'manifest.json').write_text(json.dumps(dict(kind='real_battle_bootstrap',revision=REVISION,
                records=[dict(group='same',split=s) for s in ['train','val']])))
            with patch('subprocess.check_output',return_value=REVISION):
                with self.assertRaisesRegex(ValueError,'crosses'):
                    enrich_real(base,root/'source',root/'out')
            self.assertFalse((root/'out').exists())
