"""Build a self-contained Windows executable from an explicit source directory."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path)
    parser.add_argument('--assets-from', type=Path, help='Optional cache of the pinned public baseline only')
    args = parser.parse_args()
    source = args.source.resolve()
    output = (args.output or source).resolve()
    resources = source/'build/standalone-resources'
    resources.mkdir(parents=True, exist_ok=True)
    (resources/'defaults').mkdir(exist_ok=True)
    sys.path.insert(0, str(source))
    from crbot.battlefield_assets import ASSETS, RELATIVE_ROOT, install_assets, verify_asset
    cache = (args.assets_from or source).resolve()
    for name in ASSETS:
        cached = cache / RELATIVE_ROOT / name
        if verify_asset(cached, name):
            target = resources/'defaults'/RELATIVE_ROOT/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cached, target)
    shutil.copy2(source/'THIRD_PARTY_NOTICES.md', resources/'defaults/THIRD_PARTY_NOTICES.md')
    install_assets(resources/'defaults')
    from crbot.learning_store import digest
    manifest = {p.name: hashlib.sha256(p.read_text(encoding='utf-8-sig').encode()).hexdigest()
                for p in sorted((source/'crbot').glob('*.py'))}
    (resources/'build-info.json').write_text(json.dumps({
        'code_hash': digest(manifest),
        'modules': manifest,
    }, indent=2), encoding='utf-8')
    from PIL import Image, ImageDraw, ImageFont
    image = Image.new('RGB',(620,320),'#0B1120')
    draw = ImageDraw.Draw(image)
    for y in range(320):
        draw.line((0,y,620,y),fill=(11+int(y/45),17+int(y/30),32+int(y/18)))
    font_path = Path(os.environ.get('WINDIR','C:/Windows'))/'Fonts/segoeui.ttf'
    title = ImageFont.truetype(str(font_path),36)
    note = ImageFont.truetype(str(font_path.parent/'msyh.ttc'),15)
    draw.ellipse((267,35,353,121), outline='#2B405C', width=4)
    draw.arc((267,35,353,121),-90,160,fill='#44D7E8',width=4)
    draw.text((310,78),'RL',font=title,fill='#F4F7FF',anchor='mm')
    draw.text((310,160),'Royal Lab',font=title,fill='#F4F7FF',anchor='mm')
    draw.text((310,205),'正在准备内置运行环境，请稍候',font=note,fill='#A6B7CE',anchor='mm')
    image.save(resources/'splash.png')
    icon=Image.new('RGBA',(256,256),'#0B1120');d=ImageDraw.Draw(icon)
    d.rounded_rectangle((8,8,248,248),radius=48,fill='#142137',outline='#397CF6',width=8)
    d.text((128,125),'RL',font=ImageFont.truetype(str(font_path),104),fill='#44D7E8',anchor='mm')
    icon.save(resources/'royal-lab.ico',sizes=[(16,16),(32,32),(48,48),(64,64),(128,128),(256,256)])
    git = shutil.which('git')
    gh = shutil.which('gh')
    if not git or not gh:
        raise SystemExit('Build requires portable Git and GitHub CLI for bundled synchronization.')
    git_root=Path(git).resolve().parent.parent
    if not (git_root/'mingw64').is_dir():
        raise SystemExit('Git distribution must contain cmd, mingw64 and usr directories.')
    env = dict(os.environ, ROYAL_BUILD_SOURCE=str(source), ROYAL_BUILD_RESOURCES=str(resources),
               ROYAL_BUILD_GIT=str(git_root), ROYAL_BUILD_GH=str(Path(gh).resolve()),
               PYINSTALLER_ZLIB_COMPRESSION_LEVEL='3')
    subprocess.run([sys.executable,'-m','PyInstaller','--noconfirm',
                    '--distpath',str(output),'--workpath',str(source/'build/standalone'),
                    str(source/'packaging/RoyalLab.spec')],env=env,cwd=source,check=True)


if __name__ == '__main__':
    main()
