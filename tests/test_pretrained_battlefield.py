from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from crbot import battlefield_assets as assets
from crbot.battle_perception import LaneThreat
from crbot.cards import CardCatalog
from crbot.learned_perception import LearnedBattlefieldDetector
from crbot.model_status import model_status
from crbot.pretrained_battlefield import PretrainedBattlefield, classify_side, prepare_image, restore_box, unit_card_id
from tests.test_learning import make_card


class AdapterTests(unittest.TestCase):
    def test_letterbox_roundtrip_for_both_screen_ratios_and_odd_sizes(self):
        for w,h in [(540,960),(1080,1920),(1080,2400),(541,961),(960,540)]:
            tensor,t = prepare_image(Image.new("RGB", (w,h)))
            self.assertEqual(tensor.shape,(1,3,480,352))
            self.assertEqual(tensor.dtype,np.float16)
            left,padtop,nw,nh,top,bottom,_,_ = t
            np.testing.assert_allclose(restore_box([left,padtop,left+nw,padtop+nh],t), [0,top/h,1,bottom/h])

    def test_aliases_do_not_give_spawned_units_their_parent_statistics(self):
        catalog=CardCatalog([make_card(c) for c in ["baby_dragon","mini_pekka","archers","golem","phoenix"]])
        self.assertEqual(unit_card_id("hungry_dragon",catalog),"baby_dragon")
        self.assertEqual(unit_card_id("minipekka",catalog),"mini_pekka")
        self.assertEqual(unit_card_id("archer",catalog),"archers")
        for name in ("golemite","phoenix_egg","nonexistent","giant_snowball"):
            self.assertIsNone(unit_card_id(name,catalog))

    def detector(self):
        d=PretrainedBattlefield.__new__(PretrainedBattlefield)
        d.catalog=CardCatalog([make_card("knight")]);d.config={}
        d.names={0:"knight",1:"golemite"};d.input_name="images";d.side_input="input"
        d.device="cpu";d.calls=d.gpu_calls=d.detected_units=0
        d.session=Mock();d.side_session=Mock()
        d.side_session.run.return_value=[np.array([[.01,.99]])]
        return d

    def test_invalid_low_confidence_and_unmapped_outputs_are_not_observations(self):
        d=self.detector()
        d.session.run.return_value=[np.array([[
            [80,160,120,200,.91,0], [80,160,120,200,.2,0],
            [80,160,120,200,.91,1], [80,160,120,200,.91,2],
            [80,160,120,200,.91,.5], [80,160,120,200,np.nan,0],
            [120,160,80,200,.91,0],
        ]])]
        from tests.battlefield_fixtures import scene_with_badge
        observations=d.detect(scene_with_badge(x=.325,y=.30))
        self.assertEqual(len(observations),1)
        self.assertEqual(observations[0]["card_id"],"knight")
        self.assertEqual(observations[0]["side"],-1)
        self.assertFalse(observations[0]["deployment_confirmed"])
        self.assertEqual(d.calls,1)
        self.assertEqual(d.gpu_calls,0)

    def test_baseline_enriches_world_without_promoting_a_champion(self):
        with tempfile.TemporaryDirectory() as folder:
            baseline=Mock(metadata=assets.manifest())
            baseline.detect.return_value=[dict(card_id="knight",side=-1,bbox=[.2,.4,.3,.5],confidence=.9)]
            with patch("crbot.pretrained_battlefield.PretrainedBattlefield",return_value=baseline):
                detector=LearnedBattlefieldDetector(Path(folder),CardCatalog([make_card("knight")]),{})
            self.assertTrue(detector.available)
            self.assertIsNone(detector.champion)
            self.assertFalse(detector.runtime_metadata["promoted"])
            fallback={lane:LaneThreat(lane,0,0,0,"none",()) for lane in ("left","right")}
            result=detector.detect(Image.new("RGB",(540,960),"gray"),fallback)
            self.assertEqual(result["left"].enemy_cards,("knight",))
            self.assertEqual(detector.observed_enemies[0]["x"],.25)
            baseline.detect.side_effect=RuntimeError("inference failed")
            self.assertEqual(detector.detect(Image.new("RGB",(540,960)),fallback),fallback)
            self.assertEqual(detector.observed_enemies,[])
            self.assertFalse(detector.last_detection_succeeded)

    def test_missing_assets_are_visible_and_disabled_does_not_try_loading(self):
        with tempfile.TemporaryDirectory() as folder:
            d=LearnedBattlefieldDetector(Path(folder),CardCatalog([]),{})
            self.assertFalse(d.available)
            self.assertIn("setup_battlefield",d.error)
            with patch("crbot.pretrained_battlefield.PretrainedBattlefield") as load:
                d=LearnedBattlefieldDetector(Path(folder),CardCatalog([]),{"pretrained_enabled":False})
                load.assert_not_called()
            self.assertFalse(d.available)

    def test_local_champion_takes_priority_over_the_public_baseline(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);directory=root/"models/battlefield";directory.mkdir(parents=True)
            (directory/"test.pt").write_bytes(b"fake checkpoint")
            champion=dict(version="local",model_path="test.pt",dataset_manifest={"classes":["enemy__knight"]})
            (directory/"registry.json").write_text(json.dumps(dict(champion=champion,candidates=[])))
            model=Mock()
            with patch.dict("sys.modules",{"ultralytics":SimpleNamespace(YOLO=Mock(return_value=model))}), patch("crbot.pretrained_battlefield.PretrainedBattlefield") as baseline:
                d=LearnedBattlefieldDetector(root,CardCatalog([make_card("knight")]),{})
            baseline.assert_not_called()
            self.assertIs(d.model,model)
            self.assertIsNone(d.pretrained)
            self.assertEqual(d.runtime_metadata,champion)

    def test_ui_distinguishes_loaded_baseline_from_champion(self):
        with tempfile.TemporaryDirectory() as folder:
            model=SimpleNamespace(available=True,champion=None,runtime_metadata=assets.manifest())
            summary,detail=model_status(Path(folder),running=True,policy=SimpleNamespace(learned_detector=model))
            self.assertIn(assets.VERSION,summary)
            self.assertIn("公开预训练，待本地准确率验收",detail)
            self.assertIn("当前冠军：无",detail)


class AssetTests(unittest.TestCase):
    def test_private_roundtrip_preserves_hashes_and_does_not_write_champion(self):
        payload=b"test model";digest=hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as folder, patch.dict(assets.ASSETS, {"units_M_480x352.onnx":(len(payload),digest),"side.onnx":(len(payload),digest)},clear=True):
            root=Path(folder)/"first";remote=Path(folder)/"remote";second=Path(folder)/"second"
            for name in assets.ASSETS:
                p=root/assets.RELATIVE_ROOT/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(payload)
            self.assertEqual(assets.sync_assets(root,remote)["published_files"],2)
            self.assertEqual(assets.sync_assets(second,remote)["received_files"],2)
            self.assertTrue(assets.assets_ready(second))
            self.assertFalse((second/"models/battlefield/registry.json").exists())
            self.assertFalse(json.loads((second/assets.RELATIVE_ROOT/"manifest.json").read_text())["local_quality_validated"])
            self.assertEqual(assets.sync_assets(second,remote)["published_files"],0)
            (remote/"models/battlefield_pretrained"/assets.VERSION/"side.onnx").write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError,"校验失败"):
                assets.sync_assets(Path(folder)/"third",remote)

    def test_download_hash_failure_does_not_replace_previous_file(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/assets.RELATIVE_ROOT/"units_M_480x352.onnx"
            p.parent.mkdir(parents=True);p.write_bytes(b"old")
            response=Mock();response.__enter__=Mock(return_value=response);response.__exit__=Mock(return_value=False)
            response.read.return_value=b"wrong download"
            with patch.object(assets,"urlopen",return_value=response), self.assertRaisesRegex(ValueError,"校验失败"):
                assets.install_assets(Path(folder))
            self.assertEqual(p.read_bytes(),b"old")


if __name__ == "__main__":
    unittest.main()
