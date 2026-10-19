# 队员 B：数据审计工具交接

交付日期：2026-10-20。分支：`codex/exact-duplicate-audit`。由 C 审核并集成，A 使用结论安排基线；本分支不直接向 `main` 提交，也不包含训练增强或类别不平衡效果结论。

## 本次交付

| 工具 | 作用 | 说明 |
| --- | --- | --- |
| `scripts/audit_duplicates.py` | 文件大小筛选、SHA-256、逐字节核验；支持哈希缓存恢复 | [精确查重](exact-duplicate-audit.md) |
| `scripts/pilot_image_features.py` | 小样本特征流程、缩略图、dHash 和灰度统计 | [试跑](image-feature-pilot.md) |
| `scripts/extract_image_features.py` | 全量特征、SQLite 缓存、版本校验、进度及进程锁 | [全量提取](full-image-features.md) |
| `scripts/find_near_duplicates.py` | 跨划分 dHash 候选检索、导出截断及复核预览 | [近重复校准](near-duplicate-review.md) |
| `scripts/verify_reviewed_pairs.py` | 人工复核候选的 SIFT/RANSAC 局部诊断 | 输入说明见下文 |
| `scripts/calibrate_pair_verification.py` | 已知同图变体与少量候选的有界校准 | [近重复校准](near-duplicate-review.md) |
| `scripts/profile_image_quality.py` | 读取既有特征生成数据画像、分布及少量质量预览 | [数据画像](data-profile.md) |

复用 `src/glaucoma/data.py` 的文件名映射与 CSV 读取。全部工具保持源图、标签和官方划分不变；测试使用临时生成的合成文件，不需要比赛数据。

## 已有结果与限制

- 全量解码、精确字节查重及全量特征提取已完成，数据画像初稿已生成。已有结果无需重新扫描；详细报告和逐图记录在本地 `outputs/`。
- 精确查重没有发现字节相同文件，不代表排除了近重复或患者重叠。
- dHash 初筛误报较多，局部验证的小规模校准发现裁剪候选召回和翻转验证盲区。目前不能用该流程作全量排除规则或自动删除依据。
- 质量预览中的助手观察与用户确认分开保存，尚未形成清洗决定；年龄异常没有自动修正。
- 训练增强、类别权重、过采样、训练/预测入口和策略效果实验不属于本次已完成范围。

## 环境与验证

常规数据工具使用项目已有 NumPy、Pillow、pandas、matplotlib；精确字节查重仅需标准库。局部验证、校准和对应测试还需要带 SIFT 的 OpenCV，本次沿用比赛容器自带的 OpenCV 4.10.0。不要因安装通用 requirements 而覆盖容器已有核心依赖。

容器核心版本和额外依赖分别在 `configs/environment/container-core-constraints.txt`、`configs/environment/container-extras.lock.txt`，是已验收环境的依赖记录，不是完整离线镜像或哈希锁文件。新增依赖重建方式由 C 复核；不含个人路径或凭证。

在已配置好的项目环境、仓库根目录执行：

```powershell
python -m unittest discover -s tests -v
```

已有比赛容器的运行方式：

```powershell
docker exec glaucoma_env /opt/dashuju-venv/bin/python -m unittest discover -s tests -v
```

测试覆盖精确重复/标签冲突、哈希缓存、特征恢复和并发保护、候选索引与全对全结果对照、导出截断、人工表保留、局部几何门槛、合成变体不改变源像素、按图片名对齐和训练分位数边界。

2026-10-20 提交前已在现有比赛容器重新执行上述完整测试：25 项全部通过，无跳过。此次未重跑全量数据扫描。

## 复现前需要的本地输入

常规运行说明中的 `D:\path\to\glaucoma_first` 是占位符，替换为自己的数据目录，并在已激活 Python 环境的仓库根目录运行。缓存依赖数据根目录、文件元信息和运行环境；不能直接在不同路径/版本下混用。

已有全量特征后，画像及候选检索直接复用 `outputs/image-features/`。校准读取候选预览 CSV；局部诊断还需人工记录 JSON。这些都是本地数据衍生产物，不在 Git 中。`verify_reviewed_pairs.py --reviews` 的结构示例如下，图片名须替换为实际已复核候选，不能把示例当作真实判断：

```json
{
  "reviews": [
    {
      "pair_id": "example-pair",
      "display_order": 1,
      "left_image": "TRAIN_IMAGE_NAME.jpg",
      "right_image": "VALID_IMAGE_NAME.jpg",
      "user_comment": "待进一步核实"
    }
  ]
}
```

运行例子：

```powershell
python scripts/verify_reviewed_pairs.py --data-root 'D:\path\to\glaucoma_first' --reviews outputs/near-duplicates/train-valid/user-review.json
```

## 提交边界与集成

Git 仅包含代码、合成测试、通用说明、依赖配置及汇总结论。原始数据、图片、逐图标签/年龄/预测、候选清单、特征表、SQLite 缓存、详细报告与模型权重均留在本地；个人项目记忆、机器配置过程和聊天摘记也不上传。

本分支承接 `chore/project-setup` 的基础读取与环境代码，并合入截至交付时的 `origin/main` 历史。由于基础分支尚未合入远程 `main`，对 `main` 的比较会同时包含这部分前置代码；C 应连同本次工具一起复核，或先集成基础分支。不要将整个工作目录手工拖入 GitHub，也不要强制推送或直接合并 `main`。

下一步是整理本地数据清单版本、对齐 A/C 的训练及评估接口，再独立开展温和增强和类别不平衡对照。近重复与患者重叠的限制继续保留，不阻塞基线启动。
