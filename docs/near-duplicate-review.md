# 近重复候选：先校准，再确认

日期：2026-10-17。已完成全量特征提取；本阶段仅用已有特征检索训练—验证的候选，不重新解码全部图片。

## 本次探索结果

64 位水平 dHash，探索性距离范围 0–4，训练 51,396 张、验证 5,849 张：

- 距离 0：49,262 对。
- 距离 1：205,705 对。
- 距离 2：505,936 对。
- 距离 3：955,254 对。
- 距离 4：1,569,113 对。
- 合计 3,285,270 个候选对。这是指纹相似的计数，不是近重复确认数。
- 导出上限为 100,000 对，按距离和样本顺序保留最靠前的候选，已明确记录截断。报告中的总候选计数未截断。

人工视觉检查最初几对预览，发现指纹距离为 0 仍出现明显不同的血管结构。因此当前 dHash 的区分能力不足，不能靠降低到距离 0 就直接认定重复，也不能要求成员人工审查全部数百万候选。

## 你当前要做什么

打开 `outputs/near-duplicates/train-valid/review.html` 或 `contact-sheet.jpg`，先查看少量图像对。左侧训练集，右侧验证集。预览按距离优先，并尽量避免左右重复使用同一图片；这是方法校准样本，不是随机抽样，不能据此估算总体误报率。

查看血管分叉与走向、视盘和视野边缘等可对应的结构，不要只凭颜色接近或同为左/右眼做判断。这是图像同一性复核，不进行病情诊断。颜色变化或图像压缩不必然意味着不同原图，结构不同则需要谨慎。

在本次 report.json 指定的 `review_file` 里填写：

- `review_status`：确认近重复 / 不同图 / 待进一步核实。
- `reviewer`：你的名字或 B。
- `review_notes`：判断依据，例如“主要血管分叉位置明显不同”或“多个血管交点对应，疑似同图调整亮度”。

HTML 页面仅展示，不会自动保存判定；可填写 CSV，也可在聊天中反馈后单独记录。用户 2026-10-18 在聊天中复核的前四对已保存到 `user-review.json` 和 `user-review-summary.md`；其内容优先于尚未填写的机器模板。助手的初步观察不是用户确认结果。

已存在的 review.csv 不被脚本覆盖，重跑会另写 review-template.csv；若搜索参数或特征文件变了，脚本要求换输出目录，避免旧人工判定对应上不同预览。

## 复现命令

本次助手已跑过，不必再次运行。若复现，在项目目录的 PowerShell 中执行：

```powershell
python -X utf8 -u scripts/find_near_duplicates.py --features-dir outputs/image-features --output-dir outputs/near-duplicates/train-valid --radius 4 --preview-count 20
```

该实现对 64 位码分段索引，并精确复核完整码汉明距离；合成测试与全对全结果对照一致，包括分段边界和重复哈希，不在预设距离内近似漏搜。这个保证只针对指纹空间，不保证真实近重复图一定落入距离范围。

## 下一步应如何改进

先用少量已复核的同图变体和不同图校准更强的特征/局部匹配，而不是继续扩大 dHash 距离。可以从候选的小范围二阶段验证开始；若需补提全量更强特征，应先试跑、固定环境并提供缓存，避免反复全量解码。

其余 train-test、valid-test 及划分内部尚未检索；跨划分风险尚未排除。未经核实不删图、改标签、调整官方验证集或手工改测试预测。报告和预览留在本地。

## 二阶段小规模校准（2026-10-18）

`scripts/calibrate_pair_verification.py` 使用现有比赛容器的 OpenCV，无需安装依赖。训练集按固定种子选取 8 张（每类 4 张），在最长边 1000 像素的内存副本上生成变亮度、变色、JPEG 压缩、缩放、旋转、裁剪、水平翻转，共 56 对确定的同图变体；再检查现有的 20 对预览候选。

这一步同时检查候选入口 dHash 的漏检与 SIFT 局部验证的表现。合成对照两侧的 dHash 在容器内用相同函数重算，不与不同 Pillow/NumPy 环境的全量缓存混用；原始特征表保持不变。

优先复核门槛在运行前固定为：RANSAC 内点至少 20 个、内点比例至少 0.5、两侧匹配凸包覆盖面积各至少 5%、对齐灰度相关性至少 0.75。只作为是否值得进一步人工查看的信号，不是自动重复判定。这是小样本可行性检查，尚无独立真实重复验证集，不能作为已验证的全量分类规则。

结果保存在 `outputs/near-duplicates/train-valid/calibration/report.md` 和 `report.json`，人工原始判断仍在 `user-review.json`，不被程序覆盖。原图、标签和官方划分不改变。需要复现时在项目目录的 PowerShell 运行：

```powershell
docker exec glaucoma_env /opt/dashuju-venv/bin/python -u scripts/calibrate_pair_verification.py --data-root /workspace/data
```

本轮结果：局部验证通过 48/56 对，漏检 8 对翻转；dHash 对裁剪和翻转各漏检 8/8，两阶段共同通过 39/56。20 对真实预览候选均未达到预设优先门槛。不能据此推断全量召回率，也不能升级为“无近重复”结论。暂不扩大现有方法的全量执行，先整理已有特征的数据画像；近重复盲区继续作为未解决事项保留。

## 方法参考

- [ImageHash 项目：感知哈希主要描述亮度结构](https://github.com/JohannesBuchner/imagehash/blob/master/README.rst)
- [Fast Search in Hamming Space with Multi-Index Hashing，原作者论文](https://www.cs.toronto.edu/~fleet/research/Papers/chunkingCVPR12.pdf)
