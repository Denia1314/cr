# Windows EXE 启动器

3.10.13 的 `RoyalLab.exe` 是图形启动入口。将它放在项目根目录，与
`start_ui.bat`、`config.json`、`crbot`、`tools` 文件夹一起使用，然后双击。
可以通过资源管理器为它创建桌面快捷方式。

双击后显示图形启动窗口，后台检查 Python、GPU 与界面依赖，然后直接打开
主界面，不调用 CMD 或 BAT。只有主界面完成创建并发出就绪信号后，启动窗口
才关闭。启动失败时在窗口显示原因与日志；关闭准备窗口会取消本次启动。
NVIDIA 自动准备 CUDA、CPU 回退及显式 CPU 设置继续复用 `tools/ensure_gpu.py`。
首次准备需要联网，可能下载较大的 PyTorch 包；已有环境也会检查可用性。

3.10.13 修复旧 EXE 继承打包目录中 Tcl/Tk 环境变量导致 `init.tcl` 无法找到
的问题：外部 Python 使用自己的库路径，并清除 PyInstaller 的 DLL 搜索路径
影响。`launcher.log` 记录准备过程，`ui-process.log` 记录界面进程输出，
`startup-error.log` 记录最近一次界面启动异常。日志保留在本地，不提交仓库。

这不是无需 Python 的完整离线安装包。电脑仍需要 Python 3.10+ 或项目支持的
现有 Python 环境；只复制 EXE 到另一台机器不能运行机器人。
模型、设备配置及训练数据继续使用原有目录和私有同步流程。
EXE 不包含任何训练数据、凭据或模型，也不会改变模型加载或自动部署规则。

## 构建

在 Windows 项目目录双击 `build_exe.bat`，或在终端运行它。
脚本安装 `requirements-build.txt` 中固定版本的 PyInstaller，并生成根目录
`RoyalLab.exe`。打包依赖只用于构建，不会成为机器人运行时必需依赖。
构建方式使用 PyInstaller 官方支持的 `--onefile --windowed`：
<https://www.pyinstaller.org/en/stable/usage.html>。

构建产物和临时目录已从代码仓库忽略；另一台电脑拉取代码后可自行构建。
更新机器人 Python 代码通常无需重建启动器；修改启动器本身时需重新构建。
从 3.10.12 升级到 3.10.13 必须重新构建并替换旧 `RoyalLab.exe`。
当前产物未做代码签名。

可运行 `RoyalLab.exe --check` 只检查项目位置和必要文件，退出码 0 表示齐全，
1 表示缺失；结果写到同目录 `launcher-check.log`。此检查不会启动机器人，
也不证明 GPU、模拟器连接或实战已通过验收。
