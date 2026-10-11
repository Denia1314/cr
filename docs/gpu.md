# 系统用户目录共享依赖（3.12.15）

BAT 启动器将 Python、PyTorch CUDA、训练依赖及 ONNX Runtime 安装到 `%LOCALAPPDATA%\RoyalLab\python-envs\python-3.12`，具体末级目录随 Python 大/小版本变化。该目录在项目外，删除/重新下载/更新项目不会删除依赖，也无需管理员权限。不同 Python 版本隔离；同版本的多个项目共用一套环境，文件锁避免并发 pip 安装。也可用绝对路径环境变量 `CRBOT_PYTHON_ENV` 指定共享位置。

`start_ui.bat`、`start_bot.bat` 及依赖准备使用同一个环境，`setup_gpu.bat` 使用该解释器验证。依赖文件摘要不变且检查通过时直接复用；依赖变更、损坏或不满足检查时修复。首次在这台电脑准备仍需要安装一次，升级依赖仍可能下载新版本。`CRBOT_COMPUTE_DEVICE=cpu` 或无 NVIDIA 驱动时保留 CPU 路径，不安装 CUDA PyTorch。直接用其他 Python 运行模块不会自动使用这套环境。

固定公开识别基线缓存在 `%LOCALAPPDATA%\RoyalLab\public-assets`，每次复用前验证固定大小和 SHA-256，然后复制到项目模型目录。损坏缓存会重新下载；个人训练模型、回放、配置和同步身份不放到公共缓存，仍走原有私有同步流程。

原来已经启动的独立 EXE 不受 BAT 改动影响；独立 EXE 有自己的用户目录运行缓存，仍需使用对应新构建。共享依赖可用不表示模型加载或实战验收已经完成。

# GPU acceleration

Windows launchers that request dependency setup now detect NVIDIA through `nvidia-smi` and run `tools/ensure_gpu.py` before starting the application. A fresh checkout creates its own `.venv`; missing or CPU-only PyTorch is replaced with explicitly pinned CUDA wheels. Existing working CUDA installations are retained. Every launch checks a real CUDA matrix multiplication. Installation or CUDA verification errors stop launch visibly. No NVIDIA driver detected means CPU mode with a diagnostic message; AMD/Intel GPUs are not supported by this CUDA setup. `CRBOT_COMPUTE_DEVICE=cpu` skips automatic CUDA setup. Direct Python invocations bypass this bootstrap.

Run `setup_gpu.bat` to force a manual setup/check on each NVIDIA computer. It creates the project `.venv`, installs CUDA PyTorch and the visual training dependencies, checks dependency integrity, and runs GPU policy parity and YOLO forward/backward/inference tests. Restart Royal Lab afterward; the existing launchers prefer `.venv`. The CUDA wheel source follows [PyTorch's official installation instructions](https://pytorch.org/get-started/previous-versions/).

`training.device: "auto"` selects CUDA when a real kernel succeeds, otherwise logs a CPU fallback. It applies to YOLO training, validation, automatic labels and runtime detection. Set it to `cuda:0` to require the first GPU, or `cpu` to disable it.

Everyday hand recognition also accelerates the ORB descriptor matching step, even without a usable learned champion. Descriptors are expanded into signed bits and matched by CUDA matrix multiplication, preserving exact Hamming distances and the existing ratio test. Eight templates per batch bound the temporary distance matrices. ORB extraction stays on CPU. `vision.card_match_device` can override the policy compute device selection with `cpu`, `auto`, or `cuda:N`.

The first completed hand match prints an explicit GPU execution message. `.training-sync/runtime-status.json` records `hand_matching.device` and `hand_matching.gpu_match_calls`, refreshed on each battle decision and with timing summaries; `not_executed` and zero calls mean it has not run yet. Check the PID and report timestamp before interpreting this file as live. Selecting CUDA or passing a standalone smoke test does not mean a learned model is loaded: no visual champion and an incompatible replay champion still disable those respective model paths.

Replay and imitation policy training evaluations and runtime predictions use CUDA for nearest-neighbor distance calculation, selection and weighted scores. `CRBOT_COMPUTE_DEVICE` accepts `auto` (default), `cpu`, `cuda` or `cuda:N`; explicit CUDA selection fails visibly if unavailable. A warning identifies the actual selected GPU on first use. Model tensors are cached with a 256 MiB / 128-entry limit; model arrays must remain immutable. Exact boundary ties use the existing NumPy partition selection to preserve policy behavior.

JSON parsing, image feature extraction, data splitting, report statistics and portable NPZ serialization remain on CPU. These steps are not neural network training. GPU use does not guarantee faster single-sample predictions; benchmark representative datasets before claiming a speedup. Existing quality gates and champion promotion rules still apply.

Check CUDA explicitly with `setup_gpu.bat`. This checks policy scoring and reports the GPU and PyTorch versions without training or promoting a model. New training candidate manifests record `compute_device`, so selecting automatic mode is distinguishable from actually running on CUDA. No local environment, weights or training records belong in Git.
