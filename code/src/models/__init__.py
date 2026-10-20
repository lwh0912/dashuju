"""模型构建入口（A 负责）。

统一约定：
- 输入：RGB Tensor，float32，形状 [B, 3, H, W]，预处理在 Dataset 内完成；
- 输出：二分类 logits，形状 [B]（squeeze 后），损失前不重复 sigmoid；
- 评估时 sigmoid(logit) 得到标签 1（青光眼阳性）的概率；
- 2026-10-19 赛务书面确认：允许使用开源预训练权重（ImageNet 等）。
"""
import torch
import torchvision

_BACKBONES = {
    "resnet18": torchvision.models.resnet18,
    "resnet50": torchvision.models.resnet50,
    "efficientnet_b0": torchvision.models.efficientnet_b0,
}


def build_model(name: str = "resnet18", num_classes: int = 1, pretrained: bool = False):
    """按名字构建模型。pretrained=True 使用 torchvision 官方 ImageNet 权重。"""
    if name not in _BACKBONES:
        raise ValueError(f"unknown model: {name}")
    m = _BACKBONES[name](weights="DEFAULT" if pretrained else None)
    if name.startswith("resnet"):
        m.fc = torch.nn.Linear(m.fc.in_features, num_classes)
    else:  # efficientnet_b0 等：分类头为 classifier[1]
        m.classifier[1] = torch.nn.Linear(m.classifier[1].in_features, num_classes)
    return m
