# 三人协作

## 当前阶段

`chore/project-setup` 用于项目基础配置和数据读取约定。通过 PR 检查并合并到 `main` 后，再从更新后的 `main` 创建任务分支。

| 分支 | 工作内容 |
| --- | --- |
| `feat/data-check` | 扩展数据质量检查、预处理和数据加载 |
| `feat/train-baseline` | 实现训练循环、验证过程和权重保存 |
| `feat/evaluate-submit` | 实现指标、测试预测和提交文件校验 |

分支名表示任务，不绑定某个成员的权限；每项新任务从最新 `main` 开始。

## 每次工作

1. 在 GitHub Desktop 中切换到 `main`，Fetch origin，再按需 Pull origin。
2. 从更新后的 `main` 创建工作分支。
3. 在 VS Code 中修改代码并运行相关检查。
4. 在 Desktop 中核对改动文件，填写说明并 Commit。
5. Publish branch / Push origin 后，在 GitHub 创建 PR，base 为 `main`。
6. 另一名成员检查后合并；检查代码时说明运行过哪些命令及结果。
7. 合并后同步本地 `main`，为下一项任务创建新分支。

## 共用约定

- 图片路径和标签读取统一使用 `src/glaucoma/data.py`，避免各自处理文件名大小写。
- 官方训练、验证、测试划分保持独立；原始文件不在代码中覆盖写入。
- 训练参数和可复现方法放入仓库；本机路径放入 `configs/paths.local.json`。
- 数据、模型权重和包含数据细节的报告留在本地，按赛事授权范围使用。
- 更改共用依赖或数据接口前与队友沟通。
- 提交说明写清“改了什么、为什么、如何验证”。
