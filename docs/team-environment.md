# 团队环境与现有路径适配

更新日期：2026-10-21。新队员先检查已有环境，C 对比版本和运行结果，B 合并确认过的依赖记录。终稿的从零安装步骤不重复作用于已验收环境。

## 已有环境

- 正确镜像：`crpi-4leqgt7v1jo7rycf.cn-beijing.personal.cr.aliyuncs.com/coder02026/compete:1.0`。终稿的 `crpi-4leqqt7v1jo7rycf` 有笔误，不应替换现有镜像或据此重拉。
- 已有 `glaucoma_env`，日常解释器为 `/opt/dashuju-venv/bin/python`，继承基础镜像依赖。旧 GPU 验收及 25 项合成测试通过，不代表三人已完成真实训练/推理链路。
- 代码和只读数据分别挂载到 `/workspace/code`、`/workspace/data`，报告保存在 `/workspace/code/outputs`。不为目录示例重复复制全部图片或重建同名容器。
- 终稿 `/workspace/outputs` 是另一种挂载布局。当前 OUTPUT_ROOT 使用 `/workspace/code/outputs`，入口应接受参数或环境变量；需要变更布局时由 C 验证并记录，不在训练中迁移。
- 已处理系统 Python 的 PEP 668/ensurepip 限制，不再照抄无版本的 pip install --user，也不升级 torch/torchvision。

## 日常入口

宿主 PowerShell 在 Docker Desktop 就绪后执行：

```powershell
docker start glaucoma_env
docker exec -it -e DATA_ROOT=/workspace/data -e OUTPUT_ROOT=/workspace/code/outputs glaucoma_env /bin/bash
```

容器 Bash 中：

```bash
source /opt/dashuju-venv/bin/activate
cd /workspace/code
python -m pip check
python -m unittest discover -s tests -v
```

变量仅在这次 exec 会话及子进程生效，不改变容器创建配置。现有审计工具需要显式 `--data-root "$DATA_ROOT"`，不能声称所有脚本都已自动读取环境变量。A/C 的完整训练/推理链路还需接入验收。

## 宿主 Git 与 SSH

终稿首选 Ubuntu/SSH。当前 Ubuntu 已有 Git/OpenSSH，但尚未发现其公钥；Windows 已有 SSH 文件，现有 GitHub SSH 认证未通过。先保留已验证可同步的 HTTPS 与宿主凭据管理器，不把 remote 改成不可用的 SSH，不把私钥复制到容器或仓库。

切换时先完成宿主个人密钥、公钥登记和 `ssh -T git@github.com` 身份核对，再验证仓库访问、修改 remote。已有同名密钥不可覆盖。身份/邮箱沿用真实配置，不用终稿占位符替换，不为本项目修改全局配置。

当前仓库级配置为 `core.autocrlf=input`、`pull.ff=only`、`push.default=simple`，源码行尾由 `.gitattributes` 控制。宿主命令入口不同不改变 PR、审批和 Merge commit 规范。

## 依赖锁定入口

- `configs/environment/container-core-constraints.txt`：已有核心版本约束。
- `configs/environment/container-extras.lock.txt`：已有额外依赖的精确版本。
- `requirements-extra.lock.txt`：兼容终稿命名的入口，引用上述文件，不另建一套版本。

这些是现有环境记录，不代表 C 已完成三台机器共同验收。新增或重建依赖先由 C 验证，B 经复核合并。正常容器不用重复安装；同一基础镜像的新容器已建好虚拟环境后，可用下面的入口补包：

```bash
python -m pip --python /opt/dashuju-venv/bin/python install -r /workspace/code/requirements-extra.lock.txt
/opt/dashuju-venv/bin/python -m pip check
```

完整 pip freeze、GPU 信息、镜像身份和验收记录留在本地或授权团队存储。个人卡验收通过不等于在 T4 上满足最终推理耗时。
