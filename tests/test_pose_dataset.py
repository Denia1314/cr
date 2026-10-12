import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from crbot.pose_dataset import export_pose, validate_reviews
from crbot.pose_validation import match_boxes, trial_gate, validate_holdouts


class PoseDatasetTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.records = []
        for split, color in [('train', 'red'), ('holdout', 'blue')]:
            path = self.root / f'{split}.png'
            Image.new('RGB', (200, 240), color).save(path)
            self.records.append(dict(path=path.name, group=split, split=split,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                annotation_source='manual_visual_review', card='mega-knight',
                bbox=[.4, .4, .6, .5]))
        self.classes = ['mega-knight', 'knight']
        self.base = self.root / 'base'
        for split in ('train', 'val'):
            for kind in ('images', 'labels'):
                (self.base / kind / split).mkdir(parents=True)
            Image.new('RGB', (200, 240), 'white').save(self.base / f'images/{split}/scene.jpg')
            (self.base / f'labels/{split}/scene.txt').write_text('1 0.05 0.05 0.04 0.04\n')
        self.manifest = dict(kind='real_battle_bootstrap', classes=self.classes,
            counts=dict(train=1, val=1), records=[], pose_sources=self.records)
        (self.base / 'manifest.json').write_text(json.dumps(self.manifest))
        self.reviews = self.root / 'reviews.json'
        self.reviews.write_text(json.dumps(self.records))

    def test_export_preserves_labels_and_original_validation(self):
        before = {p.relative_to(self.base): p.read_bytes() for p in self.base.rglob('*') if p.is_file()}
        result = export_pose(self.base, self.root, self.reviews, self.root / 'output', copies=3)
        self.assertEqual(result['counts'], dict(train=4, val=1))
        self.assertEqual(result['pose_unique_train_frames'], 1)
        self.assertEqual(result['pose_augmentations'], 3)
        self.assertFalse(result['pose_holdout_in_training'])
        for p, data in before.items():
            self.assertEqual((self.base / p).read_bytes(), data)
        self.assertEqual((self.root / 'output/images/val/scene.jpg').read_bytes(),
                         (self.base / 'images/val/scene.jpg').read_bytes())
        for path in (self.root / 'output/labels/train').glob('pose*.txt'):
            lines = path.read_text().splitlines()
            self.assertEqual(lines[0], '1 0.05 0.05 0.04 0.04')
            self.assertEqual(len(lines), 2)
            self.assertEqual(lines[1].split()[0], '0')
        self.assertTrue(all(r['background']=='scene.jpg' for r in result['records']))

    def test_cross_battle_and_identical_content_leaks_are_rejected(self):
        rows = [dict(r) for r in self.records]
        rows[1]['group'] = rows[0]['group']
        with self.assertRaisesRegex(ValueError, 'Battle group'):
            validate_reviews(self.root, rows, self.classes)
        rows[1] = dict(self.records[0], split='holdout', group='other')
        with self.assertRaisesRegex(ValueError, 'crosses splits'):
            validate_reviews(self.root, rows, self.classes)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            validate_reviews(self.root, self.records + [self.records[0]], self.classes)

    def test_changed_source_and_invalid_box_are_rejected(self):
        rows = [dict(r) for r in self.records]
        rows[0]['bbox'] = [0, 0, float('nan'), 1]
        with self.assertRaisesRegex(ValueError, 'box'):
            validate_reviews(self.root, rows, self.classes)
        (self.root / 'train.png').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            validate_reviews(self.root, self.records, self.classes)

    def test_existing_output_is_preserved(self):
        with self.assertRaisesRegex(ValueError, 'separate'):
            export_pose(self.base, self.root, self.reviews, self.base)
        self.assertTrue((self.base / 'images/train/scene.jpg').exists())

    def test_negatives_cannot_reuse_training_image_or_duplicate_holdout(self):
        holdout = [self.records[1]]
        validate_holdouts(self.root, holdout, self.manifest)
        with self.assertRaisesRegex(ValueError, 'isolation'):
            validate_holdouts(self.root, holdout + [dict(self.records[0], split='holdout', group='negative', bbox=[])], self.manifest)
        with self.assertRaises(ValueError):
            validate_holdouts(self.root, holdout * 2, self.manifest)


class PoseValidationTests(unittest.TestCase):
    def test_duplicate_detection_remains_false_positive(self):
        box = [0, 0, .5, .5]
        self.assertEqual(match_boxes([box], [dict(bbox=box, confidence=.95),
            dict(bbox=box, confidence=.8)]), dict(tp=1, fp=1, fn=0))
        self.assertEqual(match_boxes([], [dict(bbox=box, confidence=.9)]), dict(tp=0, fp=1, fn=0))
        self.assertEqual(match_boxes([box], []), dict(tp=0, fp=0, fn=1))

    def fixture(self):
        return dict(precision=1., recall=1., positive_frames=3, negative_frames=2,
            battle_groups=2, general_map50=.5, general_precision=.6, median_ms=30., evaluation_id='same')

    def test_trial_does_not_confer_battle_acceptance(self):
        base = dict(self.fixture(), recall=0.)
        result = trial_gate(base, self.fixture())
        self.assertTrue(result['passed'])
        self.assertEqual(result['scope'], 'limited_pose_trial')
        self.assertFalse(result['battle_acceptance'])

    def test_regression_nonfinite_or_changed_data_cannot_pass(self):
        base = dict(self.fixture(), recall=0.)
        for field, value, reason in [('recall', 0., 'pose_quality'),
                ('precision', .8, 'pose_quality'), ('general_map50', .47, 'general_map_regression'),
                ('general_precision', .57, 'general_precision_regression'),
                ('median_ms', 101., 'slower_than_10fps_inference'),
                ('negative_frames', 1, 'insufficient_independent_holdout'),
                ('evaluation_id', 'different', 'different_evaluation_data')]:
            with self.subTest(field=field):
                result = trial_gate(base, dict(self.fixture(), **{field:value}))
                self.assertFalse(result['passed'])
                self.assertIn(reason, result['reasons'])
        self.assertFalse(trial_gate(dict(base, recall=float('nan')), self.fixture())['passed'])
        self.assertFalse(trial_gate(base, dict(self.fixture(), median_ms=float('nan')))['passed'])
        unbound = self.fixture()
        unbound.pop('evaluation_id')
        self.assertFalse(trial_gate(unbound, unbound)['passed'])


if __name__ == '__main__':
    unittest.main()
