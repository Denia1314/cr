# MR0：只读访问诊断与自建测试采集

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
