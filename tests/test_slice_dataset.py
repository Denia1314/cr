import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, MagicMock

from PIL import Image

from crbot.slice_dataset import REVISION, import_slices, export_synthetic
from crbot.learned_perception import LearnedBattlefieldDetector
from crbot.cards import CardCatalog
from crbot.slice_detector import SliceBattlefield


class SliceDatasetTests(unittest.TestCase):
    def test_import_and_content_disjoint_export(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            segment = root / 'upstream/images/segment/hog-rider'
            segment.mkdir(parents=True)
            (root / 'upstream/LICENSE').write_text('MIT fixture')
            backgrounds = root / 'upstream/images/segment/backgrounds'
            backgrounds.mkdir()
            Image.new('RGB', (64, 64)).save(backgrounds / 'arena.jpg')
            for i in range(40):
                sprite = Image.new('RGBA', (20, 30), (i, 10, 50, 255))
                sprite.save(segment / f'hog-rider_{i%2}_{i:07d}.png')
            with patch('crbot.slice_dataset.subprocess.check_output', return_value=REVISION):
                imported = import_slices(root / 'upstream', root / 'slices')
            self.assertEqual(imported['count'], 40)
            self.assertEqual(imported['auxiliary_count'], 1)
            self.assertEqual(imported['classes'], ['hog-rider'])
            exported = export_synthetic(root / 'slices', root / 'yolo')
            self.assertEqual(sum(exported['counts'].values()), 40)
            self.assertFalse(set(exported['source_hashes']['train']) & set(exported['source_hashes']['val']))
            for path in (root / 'yolo/labels').rglob('*.txt'):
                fields = path.read_text().split()
                self.assertEqual(len(fields), 5)
                self.assertTrue(all(0 < float(v) <= 1 for v in fields[1:]))
            record = imported['records'][0]
            (root / 'slices' / record['path']).write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'Corrupt'):
                export_synthetic(root / 'slices', root / 'corrupt-export')

    def test_unpinned_source_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'images/segment').mkdir(parents=True)
            with patch('crbot.slice_dataset.subprocess.check_output', return_value='other'):
                with self.assertRaisesRegex(ValueError, 'revision'):
                    import_slices(root, root / 'output')

    def test_explicit_candidate_failure_reports_error_and_preserves_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            detector = LearnedBattlefieldDetector(Path(directory), CardCatalog([]),
                                                   {'slice_model_path': 'missing.pt'})
            self.assertFalse(detector.available)
            self.assertTrue(detector.error)
            fallback = {'left': object(), 'right': object()}
            self.assertIs(detector.detect(Image.new('RGB', (64, 64)), fallback), fallback)

    def test_slice_identity_requires_visible_team_badge(self):
        detector = SliceBattlefield.__new__(SliceBattlefield)
        detector.catalog = object()
        detector.metadata = {}
        detector.config, detector.device, detector.calls = {}, 'cpu', 0
        detector.model = MagicMock()
        detector.model.names = {0: 'hog-rider'}
        boxes = detector.model.predict.return_value[0].boxes
        boxes.cls.cpu.return_value.tolist.return_value = [0]
        boxes.conf.cpu.return_value.tolist.return_value = [.95]
        boxes.xyxyn.cpu.return_value.tolist.return_value = [[.1, .3, .2, .5]]
        image = Image.new('RGB', (640, 640))
        with patch('crbot.slice_detector.unit_card_id', return_value='hog_rider'), \
             patch('crbot.slice_detector.find_level_badges', return_value=[]) as level_badges, \
             patch('crbot.slice_detector.associate_badges', return_value={}) as badges:
            self.assertEqual(detector.detect(image), [])
            badges.return_value = {0: dict(side=0, confidence=.9, evidence='level_badge', level=11, bbox=[.1,.2,.2,.3])}
            observation = detector.detect(image)[0]
            self.assertEqual(observation['card_id'], 'hog_rider')
            self.assertEqual(observation['side'], 0)
            self.assertEqual(observation['confidence'], .9)
            self.assertFalse(observation['deployment_confirmed'])
            detector.metadata = {'training_dataset_kind':'real_battle_bootstrap'}
            observation = detector.detect(image)[0]
            self.assertEqual(detector.model.predict.call_args.args[0].height, 512)
            self.assertAlmostEqual(observation['bbox'][1], .24)
            self.assertAlmostEqual(observation['bbox'][3], .4)
            detector.fallback = MagicMock()
            detector.fallback.detect.return_value = [{'identity_source':'public_pretrained'}]
            badges.return_value = {}
            self.assertEqual(detector.detect(image), [])
            detector.fallback.detect.assert_not_called()
            self.assertFalse(detector.last_fallback_used)
            level_badges.return_value = [dict(side=-1, confidence=.9, evidence='level_badge')]
            self.assertEqual(detector.detect(image), [{'identity_source':'public_pretrained'}])
            detector.fallback.detect.assert_called_once_with(image, level_badges=level_badges.return_value)
            self.assertEqual(detector.last_primary_units, 0)
            self.assertTrue(detector.last_fallback_used)


class TrainingArgumentSafetyTests(unittest.TestCase):
    def test_oversized_safe_batch_rejected_before_loading_model(self):
        from crbot.slice_dataset import main
        from contextlib import redirect_stderr
        import io
        message=io.StringIO()
        with redirect_stderr(message), self.assertRaises(SystemExit) as error:
            main(['train','--safe-finetune','--batch','81'])
        self.assertEqual(error.exception.code,2)
        self.assertIn('batch',message.getvalue())
