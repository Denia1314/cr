# MR0：只读访问诊断与自建测试采集

## 3.12.14 / MR1：本地手动采集窗口

新增 `memory_capture.bat`。选择已连接的隔离测试设备，进入受控训练或友谊对局。圣水满10时点击“记录基准”，等待窗口显示基准就绪；准备立即出牌时点击“开始出牌采集”，等显示“现在可以出牌”再手动出一张牌，填写费用并点击“标记已出牌”。圣水恢复后可点击停止，或等待自动结束。窗口按钮直接控制本机程序，不需要聊天触发。关闭窗口会先发送停止并等待采集进程退出。

首先使用官方 NDK 编译：

```powershell
.\.venv\Scripts\python.exe tools\build_memory_lab.py --ndk <本机NDK目录> --kind observer --abi x86_64
```

采集器部署前验证 ABI 及哈希，窗口不连接、启动或修改模拟器，也不自动打开 Root。选择设备是明确的操作；原生产实例保持原设置。默认生成 `.memory-lab/bin/observer-x86_64`，ARM64 设备应构建对应 ABI，并通过 `--collector` 指定。命令入口为 `python -m crbot.memory_capture_ui`，可选 `--adb`、`--serial`、`--package`、`--collector`；默认从本机配置发现 ADB 和包名，不自动选择设备。

MR1 在最多512 MiB的指定匿名区域保存两次快照，保留稳定的8字节窗口（4字节对齐）。命名guest匿名RWX映射可只读扫描。点击开始后排除等待期间已变化的位置，再发现新的变化窗口并追踪120轮；每轮等待500毫秒，实际时长含读取开销。候选上限20000，超限停止加入新位置，继续跟踪已有候选并保留 `discovery_truncated` 标记。停止后记录可能不包含完整恢复过程；它仍不能作为字段验收。PID/启动时间检查不能识别同进程内所有对象重用。

候选位模式、进程信息、出牌标记和参考截图只保存在忽略的 reports 目录；缓存快照不写盘。`native/memory_observer.c` 的 `self-test`、`overflow-fixture`、`trigger-fixture` 使用自建子进程进行稳定筛选、强制超限及显式触发检查。窗口显示MR1仅表示手动研究工具已实现，游戏字段、模型加载及真实对局验收需分别验证。MR1不接入机器人决策，旧EXE没有重新打包。

MR0 是内存观测路线的第一批基础设施。它检查已有连接及应用访问条件，提供独立 Android 测试采集器与 Windows 接收器。它没有游戏字段定位、单位结构解析或对局决策接入，也没有修改已有识图和策略路径。

## 直接使用

双击根目录 `memory_probe.bat`，或使用现有 Python 环境运行：

```powershell
.\.venv\Scripts\python.exe -m crbot.memory_lab diagnose
# 主 CLI 的同等入口：
.\.venv\Scripts\python.exe -m crbot memory-lab diagnose
```

默认只使用已连接设备。优先选择配置的本地端点；无法明确选择时要求 `--serial`，不会自动连接、启动或重启模拟器。自动查找腾讯渠道包或国际服包；可用 `--package` 指定其他合法包名。

可显式指定已有工具和设备，不写进共享配置：

```powershell
.\.venv\Scripts\python.exe -m crbot.memory_lab diagnose --adb "完整的adb.exe路径" --serial "已连接设备序列号"
```

报告默认保存在 `reports/memory-lab-diagnose.json`。`--output` 只能指定当前配置目录的 `reports/` 内文件。该目录由 Git 忽略，不上传机器信息。退出码 2 表示目标不可用、权限阻断或诊断失败；详见报告里的 status。`access_open_only` 只说明文件可打开，仍不证明游戏内存字节或字段已读取。

默认诊断仅执行查询，并以只读方式打开 `/proc/PID/maps` 和 `/proc/PID/mem` 的文件描述符后关闭，不读取游戏内存。诊断前后核对 PID 与启动时间，避免把复用 PID 当同一进程。只有显式提供 `--collector` 时，才部署下述独立程序执行同样的原生访问检查。

若需要区分“文件可打开”和“实际字节可读取”，可在已经验证权限的测试实例显式增加 `--collector <ELF路径> --sample-platform-header`。该模式仅尝试读取 `/system/` 或 `/apex/` 中 libc/app_process 的平台 ELF 头 16 字节，不接受地址参数，不输出地址或字节内容。`platform_header_verified` 表示目标进程内平台模块头读取已验证，仍不表示游戏字段、战场状态或对局已验收；报告保留 `game_fields_verified=false`。

## 编译并验证测试链路

从 [Android 官方页面](https://developer.android.com/ndk/downloads) 单独取得适用于本机的 NDK，并放在 `.memory-lab/toolchain/` 或外部工具目录。工具链不随 Git 上传。构建不依赖全局修改：

```powershell
.\.venv\Scripts\python.exe tools\build_memory_lab.py --ndk "NDK解压目录" --abi x86_64
# ARM64 客体环境可改用 --abi arm64-v8a
.\.venv\Scripts\python.exe -m crbot.memory_lab self-test
```

也可在编译完成后双击 `memory_self_test.bat`。默认 50 个样本、100ms 间隔；最多 600 个样本，总设定时长不超过 60 秒。接收端有额外的超时、包大小和数量限制。

编译产物为 `.memory-lab/bin/collector-<ABI>`。这里的 ABI 是采集器所在 Android 客体的原生架构，与游戏 APK 的声明架构分别记录；不得用游戏 ARM64 声明直接选择客体 x86_64 程序的编译目标。

测试程序部署到 Android `/data/local/tmp/royal-lab-memory/fixture-<SHA256>`，部署后复核摘要。它创建自己的子进程，通过 `process_vm_readv` 读取已知变化的测试值，验证读数、会话和字节长度；还检查无效地址、跨入不可读页的非完整读取、子进程退出后的读取拒绝。没有任意地址读取命令、游戏字段解析、写入目标内存、注入、断点、Root 切换或防护绕过。

测试值的“圣水”和“四手牌”仅是已知的合成数据。协议来源固定为 `private_child_fixture`，接收器拒绝游戏来源；它不接入 BattleWorld 或动作执行。测试完成后子进程退出，小型采集程序文件保留供下次使用。

报告默认保存在 `reports/memory-lab-self-test.json`。`fixture_verified` 证明当前设备上的测试子进程读取与持续传输成立，不能证明游戏进程可读、真实字段已定位或战斗效果通过验收。`read_ms_p95` 是客体内单次读取耗时，不能当作整个识别或跨 ADB 的端到端延迟；主机与客体时钟尚未校准，报告明确标记 `end_to_end_age_verified=false`。

需要更快测试时：

```powershell
.\.venv\Scripts\python.exe -m crbot.memory_lab self-test --samples 100 --interval-ms 10 --output reports\memory-lab-fast-self-test.json
```

## 下一阶段门槛

依次区分：设备可达、目标访问文件可打开、目标实际字节可读、字段语义已验证、字段已启用、对局已验收。MR0 实现访问条件诊断、显式限定平台模块头读取验证和自建测试进程的读取链路；游戏字段及对局接入尚未实现。

当前 Root 或调试条件不成立时，应先在独立测试实例评估模拟器官方支持的权限路径。工具本身不会改变实例权限，也不会自动复制账号或安装游戏。即使 Root 条件成立，SELinux、目标属性及系统调用限制仍需实际验证。下一阶段才研究己方圣水与手牌；读到一个变化值不构成字段验收。

代码、测试和说明按项目交付工作流提交；NDK、生成的 ELF、原始观察、机器信息和本地报告留在忽略目录。MR0 的版本显示来自共享 metadata，不改变既有模型兼容性键或训练模型版本。
