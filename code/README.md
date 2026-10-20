# A 的 v1 训练基线

提供 ResNet18/ResNet50/EfficientNet-B0 二分类训练。当前使用 `_tmp` Dataset 与 sklearn 评估，尚未对接 B/C 统一接口，无正式测试推理或融合入口。保留根目录已有数据审计工具。

在仓库根目录安装额外依赖：
```bash
.venv/bin/python -m pip install -r code/requirements-extra.txt
```
基本依赖与 CPU/GPU PyTorch 安装参见根目录 docs/getting-started.md，无需重复覆盖已有 GPU 环境。

## 运行

在 code/ 目录执行，已有本地配置不要覆盖：
```bash
cp configs/baseline.yaml configs/baseline.local.yaml
# 修改 data_root、output_dir、batch_size、num_workers
../.venv/bin/python -m src.train --config configs/baseline.local.yaml --selftest
../.venv/bin/python -m src.train --config configs/baseline.local.yaml
```
Windows 用对应环境的 python 执行同样的 `-m src.train`。相对路径以 code/ 为基准；默认输出为根目录 outputs/。数据目录含 train.csv、valid.csv 及对应图片目录，示例数据路径需替换。本地覆盖 YAML 已忽略。

无数据可以只运行 --selftest；预训练默认下载 torchvision 官方权重。离线/CPU 自测时在本地配置设 pretrained=false、batch_size=2、num_workers=0，并选择独立输出目录。CPU 自动禁用 CUDA AMP；不代表 GPU 已验收。流程冒烟可加 --train-batches 1 --valid-batches 2 --epochs 2，验证子集须包含两类；该结果不是完整验证指标。不要覆盖正式实验产物。

## 预处理与产物

RGB → Pillow BILINEAR 直接 resize → 像素/255 → 配置 mean/std 标准化，无随机增强。原始 logits 为 [B,1]，训练/验证 squeeze 为 [B]，损失前不 sigmoid，评估 sigmoid 得阳性概率。当前仅实现 resize、cosine 调度和按 AUROC 选 best；其他配置值会报错。

AdamW + 可选 pos_weight=训练集 neg/pos 的 BCE + CUDA AMP + 余弦调度。train shuffle、drop_last=True，51,396 张且 batch=64 时每轮 803 次更新；valid 不丢尾批、不打乱。

- best.pt / best_auprc.pt：分别保存 AUROC/AP 最佳轮次的权重、配置及同轮指标。
- last.pt：最后一轮权重和最后一轮指标。
- best_valid_predictions.csv / best_auprc_valid_predictions.csv：对应权重的逐图验证预测，列为 Image_name、True_Label、Probability，保留 CSV 原始标识。
- history.json：逐轮汇总，AUPRC 明确采用 sklearn.average_precision_score（AP）。

checkpoint 支持模型重载，不包含优化器/调度器/scaler/RNG 状态，不支持完整断点续训。加载已有权重时先以 pretrained=False 建模，避免额外下载。预训练使用 torchvision DEFAULT；当前 ResNet18 为 IMAGENET1K_V1，其他候选的具体权重枚举需登记。

临时 Dataset 仍强制要求 True_Label，不能直接用于无标签 test。对接时应复用根目录 src/glaucoma/data.py 的记录接口，图像用 Record.path，身份用 Record.image_name。暂勿删除仍被训练入口引用的 _tmp。

## 测试与历史结果

从 code/：
```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 ../.venv/bin/python -m unittest discover -s tests -v
```
三项回归检查覆盖文件名映射及歧义、last 元数据与验证预测、拒绝未实现配置。原有 25 项测试从仓库根目录运行。

A 提供的历史记录（非本次重训）：

| 轮次 | valid AUROC | valid AP |
|---|---:|---:|
| ep8，AUROC 最佳 | 0.9663 | 0.6664 |
| ep10，AP 最佳 | 0.9555 | 0.6755 |

历史代码待追溯：history 缺少当前代码生成的 best_auprc 字段，登记批次数与 drop_last 设置不同。本分支不能倒填为历史训练 commit。旧 last.pt 元数据可能错误，代码修复不会自动修正已有产物。

任务清单、权重、原始数据、逐图结果和完整日志不入 Git。原始实验留在受控团队存储，按 SHA256 绑定代码归档和真实训练环境。GPU/T4 时延、真实 best.pt 的独立验证、正式推理和提交链路仍待验收。
