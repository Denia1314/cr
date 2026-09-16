# Windows EXE 启动器

3.10.12 的 `RoyalLab.exe` 是轻量启动入口。将它放在项目根目录，与
`start_ui.bat`、`config.json`、`crbot`、`tools` 文件夹一起使用，然后双击。
可以通过资源管理器为它创建桌面快捷方式。

启动器调用现有 `start_ui.bat`：环境准备窗口显示依赖检查、安装进度及错误，
随后打开控制台。NVIDIA 自动准备 CUDA、CPU 回退及显式设备配置仍由现有
环境流程处理。首次准备需要联网，可能下载较大的 PyTorch 包。

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
当前产物未做代码签名。

可运行 `RoyalLab.exe --check` 只检查项目位置和必要文件，退出码 0 表示齐全，
1 表示缺失；结果写到同目录 `launcher-check.log`。此检查不会启动机器人，
也不证明 GPU、模拟器连接或实战已通过验收。
