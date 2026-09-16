# Application code, CUDA runtime, and sync tools are embedded in one executable.
import os
import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

source = Path(os.environ['ROYAL_BUILD_SOURCE'])
sys.path.insert(0, str(source))
resources = Path(os.environ['ROYAL_BUILD_RESOURCES'])
datas = [(str(source/'config.json'), 'defaults'),
         (str(source/'data'), 'defaults/data'),
         (str(source/'templates'), 'defaults/templates'),
         (str(source/'templates'), 'templates'),
         (str(resources/'build-info.json'), '.'),
         (str(resources/'royal-lab.ico'), '.'),
         (str(resources/'defaults'), 'defaults'),
         (os.environ['ROYAL_BUILD_GIT'], 'vendor/git'),
         (os.environ['ROYAL_BUILD_GH'], 'vendor/gh')]
datas += collect_data_files('ultralytics')
for distribution in ('torch', 'torchvision', 'ultralytics', 'numpy', 'Pillow', 'opencv-python-headless', 'onnxruntime-gpu'):
    datas += copy_metadata(distribution)
hiddenimports = collect_submodules('crbot') + collect_submodules('ultralytics') + ['onnxruntime']
a = Analysis([str(source/'tools/exe_launcher.py')], pathex=[str(source)],
             binaries=[], datas=datas, hiddenimports=hiddenimports,
             excludes=['tensorflow','jax','IPython','notebook','pytest'],
             noarchive=False)
pyz = PYZ(a.pure)

# The outer GUI executable carries this entire folder and caches it per build.
exe = EXE(pyz, a.scripts, [], exclude_binaries=True,
          name='RoyalLab', debug=False, bootloader_ignore_signals=False,
          strip=False, upx=False, console=False, disable_windowed_traceback=False,
          icon=str(resources/'royal-lab.ico'))
collection = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name='RoyalLab')
