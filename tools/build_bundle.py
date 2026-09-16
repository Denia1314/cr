"""Wrap an onedir runtime in one Windows GUI executable with a reusable cache."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import struct
import subprocess
import zipfile

MAGIC = b'ROYALC01'
TRAILER = struct.Struct('<qq32s8s')


def compiler() -> Path:
    return Path(os.environ.get('WINDIR', 'C:/Windows')) / 'Microsoft.NET/Framework64/v4.0.30319/csc.exe'


def build_bundle(runtime: Path, output: Path, *, icon: Path | None = None) -> None:
    """Publish atomically; an interrupted build leaves the previous EXE usable."""
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.building.exe')
    command = [str(compiler()), '/nologo', '/target:winexe', '/platform:x64', '/optimize+',
               '/reference:System.IO.Compression.dll', '/reference:System.Windows.Forms.dll',
               '/reference:System.Drawing.dll', '/out:' + str(temporary)]
    if icon:
        command.append('/win32icon:' + str(icon))
    command.append(str(Path(__file__).with_name('cached_launcher.cs')))
    subprocess.run(command, check=True)
    archive = output.with_suffix('.payload.zip')
    try:
        with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=3) as bundle:
            for path in sorted(runtime.rglob('*')):
                if path.is_file():
                    bundle.write(path, path.relative_to(runtime).as_posix())
        with archive.open('rb') as stream:
            fingerprint = hashlib.file_digest(stream, 'sha256').digest()
        with temporary.open('ab') as target, archive.open('rb') as stream:
            offset = target.tell()
            shutil.copyfileobj(stream, target, 8 * 1024 * 1024)
            target.write(TRAILER.pack(offset, archive.stat().st_size, fingerprint, MAGIC))
        temporary.replace(output)
    finally:
        archive.unlink(missing_ok=True)
        temporary.unlink(missing_ok=True)
