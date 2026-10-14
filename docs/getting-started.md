# 本地准备与数据检查

当前已提供数据读取、数据检查和环境检查。完整训练、预测、指标计算尚待实现。

## 1. 安装环境

推荐 Python 3.12。Windows PowerShell 在项目目录中运行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-cu128.txt
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe scripts/check_environment.py
```

CUDA 12.8 安装源用于当前 NVIDIA 训练环境。安装包可能较大。环境检查会用随机初始化的图像模型完成小批次前向、反向和优化器更新，不下载预训练权重。

没有 NVIDIA 显卡的队友可以只安装 CPU 版本，再安装通用依赖：

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts/check_environment.py --allow-cpu
```

CPU 检查成功只表示 CPU 环境可运行，不表示 GPU 已就绪。版本依据：[PyTorch 官方安装说明](https://pytorch.org/get-started/previous-versions/#v2100)。

## 2. 配置个人数据目录

每个人从官方渠道取得数据并解压；数据保留在仓库外。复制配置模板（已有本地配置时直接编辑，不要覆盖）：

```powershell
Copy-Item configs/paths.example.json configs/paths.local.json
```

编辑 `configs/paths.local.json` 中的 `data_root`，填写本机实际路径。JSON 中推荐使用 `/`，例如 `D:/dashuju_data/glaucoma_first`。

数据目录应包含：

```text
glaucoma_first/
  train.csv
  valid.csv
  test.csv
  train/
  valid/
  test/
```

本地路径配置、数据目录、报告、压缩包和权重受 `.gitignore` 排除。Git 忽略规则不影响已经被追踪的文件，也无法自动识别任意位置的数据；提交前检查文件列表。

## 3. 运行检查

只有数据检查需求时，安装 Pillow 即可；读取模块和单元测试只依赖 Python 标准库。

```powershell
.\.venv\Scripts\python.exe -m pip install "Pillow>=10,<13"
.\.venv\Scripts\python.exe scripts/audit_dataset.py
```

也可以明确传入目录，并与原始 ZIP 的清单和文件大小进行对照：

```powershell
.\.venv\Scripts\python.exe scripts/audit_dataset.py --data-root "D:\dashuju_data\glaucoma_first" --archive "D:\dashuju_data\glaucoma_first.zip"
```

默认报告输出到 `outputs/data-audit/report.md` 和 `report.json`，不会被 Git 跟踪。检查失败时退出码非零，详细原因保留在报告中。

检查范围：

- 全量读取 CSV，核对字段、标签、重复标识、图片存在性和未标注图片。
- 全量读取图片头，统计尺寸和颜色模式；每个有标签划分分别抽取最多 32 张阳性和 32 张阴性图片完整解码，测试集抽取最多 32 张。
- 可选 ZIP 清单与解压文件大小对照；相同 CRC 和大小只用于标记重复候选。
- 不替换官方划分、不修改标签、不自动填补缺失年龄。
- 抽样解码不等于全量像素解码。ZIP 大小对照不等于全量内容 CRC 校验。
- CSV 不提供患者标识时无法判断患者重叠；未进行近似重复图片检测。

## 4. 读取数据的约定

CSV 图片字段为 `Image_name`，年龄为 `Age`；训练与验证标签为 `True_Label`。测试集没有标签。

文件名的扩展名可能存在大小写差异。`src/glaucoma/data.py` 通过目录索引找到磁盘上的真实路径，并保留 CSV 原始文件名。后续读取图片使用 `Record.path`，生成提交结果使用 `Record.image_name`；如果主办方发布新提交模板，应再次核对。

年龄缺失表示为 `None`，没有被替换为 0。后续采用年龄特征时，只用训练集估计填补和标准化参数。

## 5. 检查代码修改

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

目前测试覆盖文件名大小写映射、原始标识保留、年龄缺失、无标签测试集、非法标签、重复标识和图片缺失。

先确认官方预训练权重规则，再决定模型初始化方式。当前环境检查明确使用 `weights=None`。
