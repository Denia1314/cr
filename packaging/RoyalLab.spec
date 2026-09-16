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

class AnimatedSplash(Splash):
    def generate_script(self):
        # Runs in the bootloader's Tcl thread, so animation continues while the
        # multi-gigabyte CUDA archive is being extracted, before Python exists.
        return super().generate_script() + r'''
set royal_angle 0
.root.canvas create arc 267 35 353 121 -outline #44D7E8 -width 4 -style arc -extent 90 -tag royalspinner
proc royal_tick {} {
    global royal_angle
    if {![winfo exists .root.canvas]} {return}
    set royal_angle [expr {($royal_angle + 6) % 360}]
    .root.canvas itemconfigure royalspinner -start [expr {-$royal_angle}]
    after 25 royal_tick
}
if {![info exists ::env(CRBOT_REDUCED_MOTION)] || $::env(CRBOT_REDUCED_MOTION) ne "1"} {
    royal_tick
}
'''

splash = AnimatedSplash(str(resources/'splash.png'), binaries=a.binaries, datas=a.datas,
                        full_tk=False, minify_script=False)
exe = EXE(pyz, a.scripts, splash, splash.binaries, a.binaries, a.datas, [],
          name='RoyalLab', debug=False, bootloader_ignore_signals=False,
          strip=False, upx=False, console=False, disable_windowed_traceback=False,
          icon=str(resources/'royal-lab.ico'))
