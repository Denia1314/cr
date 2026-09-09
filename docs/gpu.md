# GPU acceleration

Run `setup_gpu.bat` once on each NVIDIA computer. It creates the project `.venv`, installs CUDA PyTorch and the visual training dependencies, checks dependency integrity, and runs GPU policy parity and YOLO forward/backward/inference tests. Restart Royal Lab afterward; the existing launchers prefer `.venv`. The CUDA wheel source follows [PyTorch's official installation instructions](https://pytorch.org/get-started/previous-versions/).

`training.device: "auto"` selects CUDA when a real kernel succeeds, otherwise logs a CPU fallback. It applies to YOLO training, validation, automatic labels and runtime detection. Set it to `cuda:0` to require the first GPU, or `cpu` to disable it.

Replay and imitation policy training evaluations and runtime predictions use CUDA for nearest-neighbor distance calculation, selection and weighted scores. `CRBOT_COMPUTE_DEVICE` accepts `auto` (default), `cpu`, `cuda` or `cuda:N`; explicit CUDA selection fails visibly if unavailable. A warning identifies the actual selected GPU on first use. Model tensors are cached with a 256 MiB / 128-entry limit; model arrays must remain immutable. Exact boundary ties use the existing NumPy partition selection to preserve policy behavior.

JSON parsing, image feature extraction, data splitting, report statistics and portable NPZ serialization remain on CPU. These steps are not neural network training. GPU use does not guarantee faster single-sample predictions; benchmark representative datasets before claiming a speedup. Existing quality gates and champion promotion rules still apply.

Check CUDA explicitly with `.venv\Scripts\python.exe -m crbot.gpu`. This checks policy scoring and reports the GPU and PyTorch versions without training or promoting a model. New training candidate manifests record `compute_device`, so selecting automatic mode is distinguishable from actually running on CUDA. No local environment, weights or training records belong in Git.
