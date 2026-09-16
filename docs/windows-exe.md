# Windows 独立 EXE

`RoyalLab.exe` 包含控制台、识别与推演、标定与标注、示范学习、自主学习、模型管理和同步模块，以及 Python、OpenCV、PyTorch/CUDA、Ultralytics、Git 和 GitHub CLI。运行不再调用外部 Python、BAT、CMD 或 VBS；`启动控制台.vbs` 已移除。

双击从解包阶段就显示动态加载环，随后淡入主界面。按钮悬停使用渐变，任务运行时状态灯呼吸。可在“工具 → 减少动态效果”关闭当前进程的动画；设置环境变量 `CRBOT_REDUCED_MOTION=1` 可从启动起关闭 Tk 动画。

## 使用和数据

只复制这个 EXE 到其他目录也能启动，无需安装 Python 或复制项目源码。CUDA 库使文件体积较大，第一次解包需要等待，临时磁盘也需有足够空间。MuMu/安卓模拟器和 NVIDIA 驱动仍由电脑安装；打包程序不能代替这些外部设备环境。

- 若 EXE 同目录已有 `config.json`，继续使用该目录，保留原来的模型、配置、标定及同步身份。
- 独立放置时，默认数据目录为 `%LOCALAPPDATA%\RoyalLab`。首次运行写入内置默认配置、公开卡牌知识和识别模板；升级不会覆盖用户已有文件。
- 可用 `RoyalLab.exe --data-dir "D:\RoyalLabData"` 指定数据目录。
- 私有训练数据、登录凭据、机器身份和自己训练的模型不写进 EXE。已有模型保留在数据目录，其他设备通过原有私有同步获取。GitHub 同步仍需要用户账号登录；内置工具不包含任何人的凭据。
- EXE 另含经过固定 SHA-256 校验的公开战场识别基线（约 30 MB）及其许可说明，首次运行即可使用。它不被标记为本地验收通过，也不会自动替换训练冠军。
- NVIDIA 电脑使用内置 CUDA 运行库；没有可用 GPU 时保留 CPU 路径，显式 CPU 设置仍有效。

## 构建

开发者在 Windows 双击 `build_exe.bat`。构建需要 Python、`requirements-build.txt` 和训练依赖，以及 Git、GitHub CLI。构建前通过 `setup_gpu.bat` 准备 CUDA 版 PyTorch，可以生成包含 CUDA 的 EXE；最终用户无需这些构建工具。

构建使用 `packaging/RoyalLab.spec` 和 `tools/build_standalone.py`。只打包应用代码、公开配置与资源、第三方运行库和同步工具，不打包 `.venv` 整个目录或任何本机模型、回放、账号数据。EXE 和构建缓存从代码仓库忽略。

代码在 EXE 内，更新代码后必须重新构建并替换 EXE。旧版“小启动器”不能升级为独立程序，需使用新版文件。当前产物未签名。

## 验证与诊断

- `RoyalLab.exe --self-test --report "D:\package-check.json"` 验证资源、功能模块、CPU/CUDA 运算与训练梯度、YOLO 无下载推理、多进程推演入口、学习子进程和同步工具；结果写入 JSON。
- `RoyalLab.exe --smoke-ui --report "D:\ui-check.json"` 打开真实界面后自动关闭，不启动设备检测、同步或战斗，记录版本选项与窗口尺寸。
- `RoyalLab.exe --cli --version` 或 `--cli <原 CLI 参数>` 使用内置命令行功能，输出保存到数据目录 `desktop.log`，不会弹控制台。
- 正常运行日志位于数据目录 `desktop.log`。自检通过不代表模型已加载、账号已登录或整场战斗验收通过。
