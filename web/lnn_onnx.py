#!/usr/bin/env python3
"""
LNN ONNX-compatible version

ONNX/WASM 변환 가능한 버전
- 동적 코드 제거
- 고정 크기 텐서
- torch.jit.script 호환
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LiquidCellONNX(nn.Module):
    """ONNX-compatible Liquid Cell"""

    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.W_in = nn.Linear(hidden_dim, hidden_dim)
        self.W_rec = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.tau_net = nn.Linear(hidden_dim * 2, 1)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        # Adaptive tau (1.0 ~ 5.0)
        combined = torch.cat([x, h], dim=-1)
        tau = 1.0 + 4.0 * torch.sigmoid(self.tau_net(combined))

        # ODE: dh/dt = (-h + f(x,h)) / tau
        f_xh = torch.tanh(self.W_in(x) + self.W_rec(h))
        dh = (-h + f_xh) / tau
        new_h = h + 0.1 * dh  # dt = 0.1
        new_h = self.norm(new_h)

        return new_h


class LNNModelONNX(nn.Module):
    """
    ONNX-exportable LNN

    Stateless version (hidden state passed as input)
    """

    def __init__(self, num_actions: int = 6):
        super().__init__()

        # Simple vision encoder
        self.vision = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 192)
        )

        # 4 Liquid layers
        self.cell0 = LiquidCellONNX(256)
        self.cell1 = LiquidCellONNX(256)
        self.cell2 = LiquidCellONNX(256)
        self.cell3 = LiquidCellONNX(256)

        # Input projection
        self.input_proj = nn.Linear(192, 256)

        # Output heads
        self.action_head = nn.Linear(256, num_actions)
        self.coord_head = nn.Sequential(
            nn.Linear(256, 2),
            nn.Sigmoid()
        )

    def forward(
        self,
        image: torch.Tensor,      # [1, 3, 224, 224]
        h0: torch.Tensor,         # [1, 256]
        h1: torch.Tensor,         # [1, 256]
        h2: torch.Tensor,         # [1, 256]
        h3: torch.Tensor          # [1, 256]
    ) -> tuple:
        """
        Returns: (action_logits, coords, new_h0, new_h1, new_h2, new_h3)
        """
        # Vision
        vis = self.vision(image)
        x = self.input_proj(vis)

        # Liquid layers
        h0 = self.cell0(x, h0)
        h1 = self.cell1(h0, h1)
        h2 = self.cell2(h1, h2)
        h3 = self.cell3(h2, h3)

        # Outputs
        action_logits = self.action_head(h3)
        coords = self.coord_head(h3)

        return action_logits, coords, h0, h1, h2, h3


def export_onnx():
    """Export to ONNX"""
    model = LNNModelONNX(num_actions=6)
    model.eval()

    # Inputs
    image = torch.randn(1, 3, 224, 224)
    h0 = torch.zeros(1, 256)
    h1 = torch.zeros(1, 256)
    h2 = torch.zeros(1, 256)
    h3 = torch.zeros(1, 256)

    # Export
    torch.onnx.export(
        model,
        (image, h0, h1, h2, h3),
        "lnn_model.onnx",
        input_names=["image", "h0", "h1", "h2", "h3"],
        output_names=["action_logits", "coords", "new_h0", "new_h1", "new_h2", "new_h3"],
        opset_version=14,
        do_constant_folding=True
    )

    import os
    size = os.path.getsize("lnn_model.onnx")
    print(f"Exported: lnn_model.onnx ({size/1024:.1f} KB)")

    # Count params
    params = sum(p.numel() for p in model.parameters())
    print(f"Params: {params:,} ({params/1e6:.2f}M)")


if __name__ == "__main__":
    export_onnx()
