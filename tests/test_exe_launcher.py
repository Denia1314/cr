import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from crbot import app_paths

class PackagedPathsTests(unittest.TestCase):
    def test_fresh_exe_uses_user_data_not_extraction_dir(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(app_paths.sys,'frozen',True,create=True), patch.object(app_paths.sys,'executable',str(Path(tmp)/'app/RoyalLab.exe')), patch.dict(os.environ,{'LOCALAPPDATA':tmp},clear=True):
            self.assertEqual(app_paths.data_root(),Path(tmp)/'RoyalLab')

    def test_existing_portable_config_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(app_paths.sys,'frozen',True,create=True), patch.object(app_paths.sys,'executable',str(Path(tmp)/'RoyalLab.exe')), patch.dict(os.environ,{},clear=True):
            (Path(tmp)/'config.json').write_text('{}')
            self.assertEqual(app_paths.data_root(),Path(tmp))

    def test_seeding_never_overwrites_user_config_or_calibration(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp); seed=base/'bundle/defaults'; seed.mkdir(parents=True)
            (seed/'config.json').write_text('{"default":true}')
            (seed/'data').mkdir();(seed/'data/cards.json').write_text('{}')
            with patch.object(app_paths,'frozen',return_value=True),patch.object(app_paths,'resource_root',return_value=base/'bundle'),patch.dict(os.environ,{'CRBOT_DATA_DIR':str(base/'user')}):
                root=app_paths.initialize_data(); (root/'config.json').write_text('{"user":true}')
                app_paths.initialize_data()
                self.assertEqual(json.loads((root/'config.json').read_text()),{'user':True})
                self.assertTrue((root/'data/cards.json').is_file())

    def test_frozen_learning_spawns_same_exe_without_python_m_flags(self):
        with patch.object(app_paths,'frozen',return_value=True),patch.object(app_paths.sys,'executable','RoyalLab.exe'):
            command=app_paths.learning_command(Path('user'),Path('request.json'),Path('worker.log'))
            self.assertEqual(command[0],'RoyalLab.exe')
            self.assertIn('--self-learning',command)
            self.assertIn('--worker-log',command)
            self.assertNotIn('-m',command)

    def test_frozen_code_identity_uses_packaged_manifest(self):
        from crbot.self_learning import code_identity
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'build-info.json').write_text('{"code_hash":"release-specific-hash"}')
            with patch.object(app_paths,'frozen',return_value=True),patch.object(app_paths,'resource_root',return_value=root):
                self.assertEqual(code_identity(root/'no-source'),'release-specific-hash')

if __name__=='__main__':unittest.main()
