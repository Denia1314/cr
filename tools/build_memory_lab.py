"""Build the private Android fixture with a separately installed official NDK."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess

TARGETS = {"x86_64": "x86_64-linux-android24", "arm64-v8a": "aarch64-linux-android24"}


def build(ndk: Path, abi: str, root: Path) -> Path:
    hosts = ("windows-x86_64",) if os.name == "nt" else ("linux-x86_64", "darwin-x86_64")
    prebuilt = ndk / "toolchains/llvm/prebuilt"
    toolchain = next((prebuilt / host for host in hosts if (prebuilt / host).is_dir()), None)
    if toolchain is None:
        raise ValueError("未找到适用于本机的官方 NDK LLVM 工具链")
    compiler = toolchain / "bin" / ("clang.exe" if os.name == "nt" else "clang")
    output = root / ".memory-lab/bin" / ("collector-" + abi)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run([
        str(compiler), "--target=" + TARGETS[abi], "--sysroot=" + str(toolchain / "sysroot"),
        "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-fPIE", "-pie",
        "-Wl,-z,max-page-size=16384", str(root / "native/memory_lab.c"), "-o", str(output),
    ], check=False, timeout=120)
    if result.returncode:
        raise ValueError("Android 采集器编译失败")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ndk", type=Path, required=True)
    parser.add_argument("--abi", choices=tuple(TARGETS), default="x86_64")
    args = parser.parse_args()
    print(build(args.ndk.resolve(), args.abi, Path(__file__).resolve().parents[1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
