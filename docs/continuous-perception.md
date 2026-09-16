# 连续画面与基础识别

截图、基础识别、决策现在分别运行。`automation.latest_frame_capture` 与
`automation.continuous_perception` 开启时，后台持续识别最新截图的手牌和战场等级标记。
推演、出牌确认、日志写入期间不再停止这些基础识别；每层只保留最新结果，慢消费者跳过旧帧。
控制台预览直接读取截图流，33 毫秒轮询一次，不再等待策略循环完成。

默认 `timing.capture_interval_s` 为 1/30 秒。这是采集目标，不是保证达到的帧率。
完整单位模型、时序状态更新、战场推演仍由决策循环执行；基础识别 FPS 不代表完整决策 FPS。
保持 G1 已实现阶段、策略版本和模型兼容身份不变，不表示新增模型或完成实战验收。

基础识别结果绑定原图对象与采集序号，策略只复用完全对应的结果。
手牌 ORB/CUDA 状态由锁保护，确认出牌与后台识别不能并发改写缓存。
出牌前仍复核新画面；点击后的确认仍要求截图起始时间晚于点击，不用旧识别结果充当成功证据。
识别异常会传回主循环，不能继续使用上次结果。关闭时先结束识别，再关闭采集和 MuMu IPC。

GPU 匹配先在相似度矩阵中选两个最佳项，再计算 Hamming 距离，避免转换完整矩阵。
判定比例仍使用 float64，CPU 路径和阈值不变。

运行状态 `.training-sync/runtime-status.json` 分开记录：

- `capture`：真实截图 FPS、后端和画面年龄。
- `perception_stream`：基础识别 FPS、耗时、画面年龄和跳过的截图数。
- `hand_matching`：真实 GPU 调用数，预热不计入运行调用。

读取当前模拟器画面做基准（不启动游戏、不点击、不开始对战）：

```powershell
.\.venv\Scripts\python.exe tools\benchmark_perception.py --seconds 15
```

现场采集帧率、基础识别帧率和结果年龄需要一起检查。静止菜单、录制画面回放、
动态战斗及同时运行其他 GPU 任务的结果不能混为一谈。关闭
`automation.continuous_perception` 可恢复同步识别，截图流与独立预览仍可保留。
