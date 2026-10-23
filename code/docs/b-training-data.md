# B：数据、增强与不平衡模块交接

更新日期：2026-10-22。状态：模块与 A 训练循环的小规模集成验证通过；正式效果对照尚未运行。B 负责本交付，A/C 复核后由 B 合并。仓库根目录对应终稿中的 code/，本交付不再套一层 code/，不搬动已验证的审计模块。

## 已完成的接口

| 位置 | 职责 |
| --- | --- |
| `src/data/dataset.py` | GlaucomaDataset，复用原 load_records；官方标识不变，读取实际大小写路径 |
| `src/data/transforms.py` | 确定性预处理及可选温和训练增强 |
| `src/data/sampling.py` | 训练子集类别统计、互斥损失/采样策略、worker 随机种子 |
| `src/data/pipeline.py` | 一次创建 train/valid DataLoader、损失和可记录的参数/身份摘要 |
| `src/losses/focal.py` | 稳定的单 logit Focal Loss，默认 gamma=2，无额外 alpha 类别权重 |
| `tools/analyze_errors.py` | 按官方验证身份/标签核对预测，导出 FN/FP 和指标，可接已有质量特征 |
| `configs/b-data-experiments.json` | 六组单因素候选配置，明确尚无正式效果结论 |
| `tools/prepare_b_experiments.py` | 将候选配置输出到本地目录，不启动训练 |
| `tools/check_training_integration.py` | 使用 A 本地代码快照验证接口、参数更新、权重重载和预测 CSV |

训练/验证样本返回 `image`（float32 RGB，CHW）、`label`（0/1）、`image_name`（官方 CSV 原拼写）；DataLoader 后为 BCHW、[B] 标签和名字列表。测试样本不包含 label，不能伪造标签用于评估。Age 不参与当前模型，也未填补或改写。

`GlaucomaDataset(root, split, config=TransformConfig(...), indices=None)` 可接当前 split 内的整数位置列表，拒绝重复或越界位置。权重按实际选中的训练记录计算。这个接口没有实现患者分组或 CV 划分；C 提供统一方案后再对接，不能把普通子集称为患者级划分。

## 预处理与策略

无增强时和 A PR #2 一致：PIL 转 RGB → PIL bilinear resize 到 224×224 → float32 除以 255 → ImageNet mean/std。实测四张真实图片与 A 临时 Dataset 输出逐元素完全相等。均值/标准差沿用配置，不使用验证/测试数据拟合。

默认不增强。B1 单独启用水平翻转概率 0.5、旋转 ±5°、亮度/对比度/饱和度各 0.1；默认不裁剪、不垂直翻转、不统一提亮。可选裁剪保留每个方向至少 95%，不能套用正方形回退强裁剪。验证和测试若请求增强会报错。温和范围是实验起点，不是医学有效性或效果提升的证明。

| 策略 | 唯一变化 | 预算注意事项 |
| --- | --- | --- |
| B0 `bce` | 普通 BCE，无增强 | 对照 |
| B1 `bce` + augment | 仅温和增强 | 与 B0 相同设置 |
| B2 `pos_weight` | BCE 阳性权重=训练子集阴性/阳性 | 与 A 当前默认基线对应 |
| B3 `oversample` | 按类别逆频率有放回采样，期望两类各半 | 每轮抽取次数仍等于训练子集大小，不保证实际每轮恰好半数阳性 |
| B4 `focal` | 无额外类别权重的 Focal Loss | 后续候选，不和加权/过采样自动叠加 |
| B5 `undersample` | 保留全部阳性，默认最多 10 倍阳性的阴性，无放回 | 每轮样本/步数会减少；先做前四组，不能以相同 epoch 声称预算相同 |

策略只能选一个；与旧 `pos_weight` 开关冲突时报错。只传 A 的旧配置时仍按其默认加权 BCE 行为处理。独立传入训练子集后，权重不会偷用完整训练集或验证集类别数。

配置中的 batch=64、worker=8、10 epoch 和预训练 ResNet18 源于 A 的候选配置，尚未在本机按该完整预算验收。实际验证使用 batch=8、随机初始化和一次更新，无预训练下载。正式开始前需统一可行的 batch/有效 batch 和预算；若调整，所有对应对照一起调整。

## A 的接入修改

已读取 A PR #2 的 `de53102644c166c484c1c4bece99ea09cf7a8643`。A 当前使用临时 Dataset，并从 `.df` 算类别权重。复核集成时将 A 的模型和入口放到共用根目录 `src/`；旧审计 `src/glaucoma/` 保留兼容。A 的 PR 尚未合并，本分支不复制整套 A 代码，也不代表已审核其所有模型/训练行为。

在 A `train(cfg)` 的 set_seed 和 device 设置之后，以以下内容替换整个临时 Dataset/DataLoader 构建段：

```python
from src.data.pipeline import build_training_bundle

bundle = build_training_bundle(cfg)
train_ld, valid_ld = bundle.train_loader, bundle.valid_loader
cfg["data_pipeline"] = bundle.metadata  # 随 A 的 checkpoint/config 保存
```

保留 A 的模型创建、优化器、训练/评估循环。删除原 `.df` 类别统计、pos_weight 和 criterion 构建段，替换为：

```python
criterion = bundle.criterion.to(device)
```

不能只替换 Dataset 类名而保留 `.df` 访问。采样器已由 bundle 配好，不额外设置 shuffle=True。A/C 的正式评估统一前，集成自测沿用 A 的 `roc_auc_score` 和 `average_precision_score`；不能称为 C 已验收的统一评估。

在容器仓库根目录生成后续候选配置：

```bash
python tools/prepare_b_experiments.py --data-root /workspace/data --output-dir outputs/b-experiment-configs
```

JSON 配置可由 A 现有 YAML 读取器读取，但必须先完成上述接入；直接用未修改的 A 入口会忽略 data_strategy/augmentation，不能用来做本组对照。生成器拒绝覆盖非空目录。

## 验证结果与复现

- 比赛容器 35 项单元测试通过（原 25 项 + 新 10 项），含无增强像素等价、训练子集权重、验证/测试保护、采样概率与预算、两 worker 复现、极端 logits 梯度、身份/标签/非有限预测拒绝。
- 真实数据只选训练 32、验证 32、测试 4 张作集成输入；副本与原图 68 个 SHA-256 全部一致。训练/验证刻意含两类，类别比例不是原始分布。
- 六种设置分别经过 A 的 run_epoch 和 evaluate；每种仅执行一个 batch（8 次训练抽取），参数实际更新、loss 有限。B0 另验证 checkpoint 重载后一致，生成四行模型预测 CSV。该 CSV 不完整且模型未正式训练，文件明确标注 NOT_FOR_SUBMISSION。
- 小样本训练阴阳比为 3:1，低于 B5 默认 10:1 上限，因此这轮 B5 不减少样本；实际减少阴性、保留阳性和无放回行为由合成测试覆盖。
- 四张无增强真实图片的 A/B 预处理完全一致；脚本检查运行前后源码哈希不变。小样本指标只用于接口自测，不用于方案排名、临床判断或正式消融结论。
- 原图、标签、Age、官方划分未改；未重复全量解码/特征提取/近重复扫描，未安装依赖。首次全目录读取较慢，已中止并改用少量本地副本；最终报告另存，不覆盖先前记录。

合成测试命令：

```bash
docker exec glaucoma_env /opt/dashuju-venv/bin/python -m unittest discover -s tests -v
```

集成脚本需 A 的可信本地代码快照及数据目录。它会先读 CSV/目录身份，再选 32/32/4 小样本；Windows 挂载盘可预先在本地生成保留官方名字与标签的少量原图副本，避免多次枚举全目录。不得把副本目录当作完整数据版本。

```bash
python tools/check_training_integration.py --data-root /path/to/local-smoke-data --a-code-root /path/to/a-code-snapshot --output-dir outputs/b-integration-smoke
```

## 错误分析与下一步

正式验证预测需要 `Image_name,True_Label,Probability` 三列。默认要求完整官方验证集；只分析小样本必须显式传 `--allow-subset`，报告保留 full_coverage=false。阈值使用 `probability >= threshold`，不会影响同一连续概率的 AUROC/AP。

```bash
python tools/analyze_errors.py --data-root /workspace/data --predictions outputs/experiment/best_valid_predictions.csv --output-dir outputs/error-analysis --features outputs/image-features/features.csv
```

输出 `fn.csv`、`fp.csv` 与 report.json 仅放本地；按概率排序供人工复核，质量分组只是描述性统计，不据此自动删图、改标签或判断病情。已用本轮小样本预测验证工具链，真实失败模式总结要等稳定模型的正式验证预测。

后续顺序：A/C 复核本交付与接口 → B 集成已经通过复核的 A/B/C 代码 → 先复现 A 的 B2 加权基线，再以 B0 为统一对照运行 B1/B2/B3 → 保存配置、种子、数据/代码版本和全验证指标，C 复核评估口径。预训练权重、正式预算及 C 评估尚未在本轮共同验收，不能把接口自测写成这一步已完成。
