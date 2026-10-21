# 三人协作与 Git 集成

## 当前阶段

更新日期：2026-10-22。采用团队指定的《三人分工与环境配置》终稿，替代此前由 C 集成的安排。已有仓库不重新初始化，不改写已发布历史。

- A：模型、训练、融合与最终赛事提交。
- B：数据/Dataset、增强、采样/损失、错误分析；组织复核并统一执行 main 合并。
- C：划分、指标、推理、CSV 校验、性能工具；环境、依赖与最终复现验收。
- B 自己的改动由 A/C 至少一人复核，之后再由 B 合并。作者自测或助手检查不能冒充队友批准。

仓库按用户确认暂时保持公开，仅上传代码、配置模板和通用说明。数据集、真实划分清单、逐图结果、完整运行记录、权重及个人信息留在本地或赛事允许的团队存储；终稿的私有仓库建议不改变本次明确选择。

当前交付包含基础配置、数据读取和 B 的审计工具及数据版本说明。后续从最新 main 开独立功能分支，不复用已交付分支堆叠训练实验。

| 分支 | 工作内容 |
| --- | --- |
| `feat/data-check` | 扩展数据质量检查、预处理和数据加载 |
| `feat/train-baseline` | 实现训练循环、验证过程和权重保存 |
| `feat/evaluate-submit` | 实现指标、测试预测和提交文件校验 |

分支名表示任务，不绑定某个成员的权限；每项新任务从最新 `main` 开始。

## 每次工作

Git 在宿主机运行，训练/验收在容器里运行。终稿首选 Ubuntu/SSH；现有 PowerShell/HTTPS 作为已验证的过渡路线，见 [团队环境说明](team-environment.md)。

1. 检查 `git status`；保存未提交工作后 `git fetch origin`、`git switch main`、`git pull --ff-only origin main`。本机无 main 时先 `git switch --track origin/main`，不使用 reset --hard。
2. `git switch -c codex/b-data-augment`，按任务更换分支名。A/C 可使用 feature/a-model-baseline、feature/c-eval-infer 等，示例不是固定名称。
3. 实现并做相关导入、配置、小批次前向/评估或脚本验证；文档改动检查链接/格式。
4. 查看 `git diff`，只 `git add` 本次实际文件，再检查 `git diff --cached` 和 `git diff --cached --check`，提交后 `git push -u origin HEAD`。
5. 创建面向 main 的 PR，写明行为、接口、依赖、真实自测结果及限制；至少一位非作者复核。B 的 PR 必须由 A/C 复核。
6. B 确认复核及检查通过，选择 **Create a merge commit**。不使用 squash/rebase merge，不绕过审批直接推 main。
7. 合并后同步 main 并记录 commit，新任务开新分支。未合并分支需要更新时由作者 `git merge origin/main`，业务冲突与 B 共同处理后再验证。

2026-10-20 的直接快进集成是当时单次授权的历史操作，不回滚重演，也不作为后续规范。

## main 保护及验收

2026-10-21 已在 GitHub 启用并读取确认：要求 PR、至少 1 个批准、更新后旧批准失效、最后一次推送由其他人批准、管理员同样受约束、讨论须解决；禁止强推和删除 main，只保留 Merge commit。

这些设置强制审批，但个人仓库未配置“仅 B 可点击合并”的人员限制；由 B 统一合并仍是团队操作约定。目前也未配置必需的 GitHub CI 状态检查。PR 须附真实检查证据，由 C 后续提供自动检查再接入，不把未运行的 CI 写成通过。

B 现有工具可在比赛容器运行：

```bash
docker exec glaucoma_env /opt/dashuju-venv/bin/python -m unittest discover -s tests -v
```

缺 OpenCV 会导致部分本机测试跳过，应如实记录并在比赛容器补齐。文档改动无需重跑全量数据扫描。

## 共用约定

- 图片路径和标签读取统一使用 `src/glaucoma/data.py`，避免各自处理文件名大小写。
- 官方训练、验证、测试划分保持独立；原始文件不在代码中覆盖写入。
- 训练参数和可复现方法放入仓库；本机路径放入 `configs/paths.local.json`。
- 数据、模型权重和包含数据细节的报告留在本地，按赛事授权范围使用。
- 更改共用依赖或数据接口前与队友沟通。
- 提交说明写清“改了什么、为什么、如何验证”。

- Dataset 提供 RGB Tensor，模型输出与损失匹配的 logits；BCEWithLogitsLoss 前不重复 sigmoid。评估/推理统一转为标签 1 的概率。训练增强和验证预处理分开，参数随配置/权重保存。
- 不为终稿 code/ 目录示例立即搬动已验证模块。A 的 PR #2 使用嵌套 code/，接入路径与兼容策略由 A/B/C 在该 PR 复核时对齐。
- 真实 splits 也含样本标识，和完整日志、逐图预测一样只放本地或赛事允许的受控存储。
- 三人启动计算前先认领实验，每天简短同步完成、阻塞和下一项；使用 [实验登记模板](experiment-record-template.md) 保留版本、预算、GPU 与指标。B 集中处理合并，不代替 A 的模型决策或 C 的验证职责。
- 终稿较早的预训练、评分及日期解读不覆盖后续赛方答疑，未决事项继续按数据版本说明记录。
