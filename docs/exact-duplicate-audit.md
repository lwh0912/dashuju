# 图片字节内容查重

队员 B 的数据质量任务；原图、标签和官方划分不改动。代码从 `codex/exact-duplicate-audit` 交付；后续由 A/C 至少一人复核 B 的改动，再由 B 统一合并。范围及当前进度见 [数据审计交接](data-audit-handoff.md)。

## 方法与边界

1. 共用 `src/glaucoma/data.py` 解析 CSV 和图片路径，处理扩展名大小写。
2. 统计所有文件大小。大小唯一的文件不可能与其他文件字节完全一致，无需读取其内容。
3. 对相同大小的候选计算 SHA-256。
4. 对相同大小且相同 SHA-256 的候选逐字节比较，形成确认组。
5. 分别统计划分内部、train-valid、train-test、valid-test 重复；比较有标签样本是否发生标签冲突。

“没有精确重复”不代表没有同一张图的重编码、缩放、裁剪版本，也不代表不存在同一患者；近重复和患者信息检查需要另行进行。

## 运行与恢复

在 Windows PowerShell 中执行：

```powershell
python -X utf8 -u scripts/audit_duplicates.py --data-root 'D:\path\to\glaucoma_first' --output-dir outputs/exact-duplicates --workers 4
```

此脚本只用标准库，直接访问 Windows 数据目录可减少跨挂载元数据开销。需先在 PowerShell 切到项目目录。实际本次扫描采用这条本机命令；哈希算法和容器版本相同，均不修改数据。与容器相比，缓存中的数据根目录不同，不跨两种路径复用哈希缓存。

若要在容器运行，也支持：

```powershell
docker exec glaucoma_env /opt/dashuju-venv/bin/python -u scripts/audit_duplicates.py --data-root /workspace/data --output-dir outputs/exact-duplicates --workers 4
```

同一输出目录只运行一个任务。如果已有任务正在运行，不再启动第二份。

扫描按阶段输出进度，摘要写入 `outputs/exact-duplicates/progress.json`。CSV/目录解析本身也需一定时间。SHA-256 阶段每 100 个结果或约 5 秒更新一次（若当前读文件较慢，需等读取返回）。

中断后用相同命令恢复。`hash-cache.jsonl` 按数据根目录、相对路径、大小和修改时间复用成功哈希；文件大小或修改时间变化时重新计算。恢复仍要重新核对目录和大小，但无需重复读取已缓存候选的全部内容。原始数据应保持不变；若怀疑内容被修改但元信息被保留，改用新输出目录重新扫描。

## 结果

- `report.md`：便于阅读的结论。
- `report.json`：完整状态、计数、分组及错误。
- `duplicates.csv`：确认重复的逐图清单，即便无重复也保留表头。
- `hash-cache.jsonl`：恢复扫描用缓存。
- `progress.json`：当前阶段和计数。

这些文件包含数据细节，默认留在 Git 忽略的 outputs 目录。代码、通用操作说明和合成数据测试可由队友复核。

只有 `status=COMPLETE` 且错误列表为空才能宣称全量精确查重完成。同一组可能同时属于多个范围，因此范围表不能直接相加作为总组数。冗余副本数等于每组数量减一之和，属于统计指标，不是自动删除建议。

## 测试

```powershell
docker exec glaucoma_env /opt/dashuju-venv/bin/python -m unittest discover -s tests -v
```

新增合成数据测试覆盖：同名大小写映射下的读取、不同文件名的相同内容、相同大小但不同内容、划分内与跨划分重复、标签冲突、哈希恢复以及同大小文件变更后的缓存失效。既有数据读取测试同时运行。
