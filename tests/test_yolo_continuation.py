import json
import hashlib
import tempfile
from pathlib import Path
import unittest
from unittest.mock import Mock

from crbot.yolo_continuation import choose_start, dataset_fingerprint, latest_completed_model


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.source=self.root/'data';self.output=self.root/'runs'
        self.source.mkdir();(self.source/'labels/train').mkdir(parents=True)
        (self.source/'manifest.json').write_text(json.dumps({'classes':['knight']}))
        (self.source/'data.yaml').write_text('names: [knight]')
        self.info=dict(epoch=3,epochs=100,optimizer=True,classes=['knight'],data=str(self.source/'data.yaml'))
        self.inspect=Mock(return_value=self.info)

    def checkpoint(self,name='candidate'):
        p=self.output/name/'weights/last.pt';p.parent.mkdir(parents=True);p.write_bytes(b'fixture')
        return p

    def test_interrupted_run_resumes_and_empty_new_directory_is_ignored(self):
        p=self.checkpoint();(self.output/'candidate-2').mkdir()
        plan=choose_start(self.source,self.output,'base.pt',inspect=self.inspect)
        self.assertEqual(plan['mode'],'resume');self.assertEqual(plan['model'],str(p))
        self.assertEqual(plan['completed_epochs'],4)

    def test_completed_run_uses_best_weights_for_finetune(self):
        p=self.checkpoint();best=p.parent/'best.pt';best.write_bytes(b'best')
        (p.parent.parent/'results.csv').write_text('epoch,loss\n100,0.1\n')
        self.info.update(epoch=-1,optimizer=False)
        plan=choose_start(self.source,self.output,'base.pt',inspect=self.inspect)
        self.assertEqual(plan['mode'],'finetune');self.assertEqual(plan['model'],str(best))
        self.assertEqual(plan['completed_epochs'],100)

    def test_changed_data_and_class_mismatch_require_explicit_choice(self):
        p=self.checkpoint();state=p.parent.parent/'training-state.json'
        state.write_text(json.dumps({'dataset_fingerprint':dataset_fingerprint(self.source)}))
        (self.source/'labels/train/a.txt').write_text('0 .5 .5 .1 .1')
        plan=choose_start(self.source,self.output,'base.pt',inspect=self.inspect)
        self.assertEqual(plan['mode'],'finetune')
        self.assertTrue(plan['dataset_changed'])
        self.info['classes']=['other']
        with self.assertRaisesRegex(ValueError,'classes differ'):
            choose_start(self.source,self.output,'base.pt',inspect=self.inspect)
        self.assertEqual(choose_start(self.source,self.output,'base.pt',fresh=True)['mode'],'new')

    def test_corrupt_completed_best_is_not_silently_replaced_by_old_model(self):
        p=self.checkpoint();(p.parent/'best.pt').write_bytes(b'changed')
        (p.parent.parent/'manifest.json').write_text(json.dumps({'model_sha256':'incorrect'}))
        self.info.update(epoch=-1,optimizer=False)
        with self.assertRaisesRegex(ValueError,'checksum'):
            choose_start(self.source,self.output,'base.pt',inspect=self.inspect)

    def test_runtime_ignores_unfinished_run_and_chooses_latest_completed_family(self):
        anchor=self.checkpoint();old=anchor.parent/'best.pt';old.write_bytes(b'old')
        meta=dict(classes=['knight'],revision='pin',kind='external_slice_candidate',model_sha256=hashlib.sha256(b'old').hexdigest())
        (old.parent.parent/'manifest.json').write_text(json.dumps(meta))
        new=self.checkpoint('candidate-2').parent/'best.pt';new.write_bytes(b'new')
        self.assertEqual(latest_completed_model(old),old)
        meta['model_sha256']=hashlib.sha256(b'new').hexdigest()
        (new.parent.parent/'manifest.json').write_text(json.dumps(meta))
        self.assertEqual(latest_completed_model(old),new)
        new.write_bytes(b'bad')
        with self.assertRaisesRegex(ValueError,'checksum'):
            latest_completed_model(old)
