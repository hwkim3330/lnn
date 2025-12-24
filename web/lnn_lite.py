#!/usr/bin/env python3
"""
LNN Lite - 브라우저용 경량 Liquid Neural Network

특징:
- ~5M params (~10MB FP16)
- MobileViT 또는 ViT-Tiny 호환
- ONNX/WASM 변환 가능
- WebGPU/WebNN에서 실행 가능
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import math


class LiquidCellLite(nn.Module):
    """
    경량 Liquid Cell

    ODE: dx/dt = (-x + f(x, I)) / τ
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        dt: float = 0.1,
        tau_min: float = 1.0,
        tau_max: float = 5.0
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dt = dt
        self.tau_min = tau_min
        self.tau_max = tau_max

        # Input transformation
        self.W_in = nn.Linear(hidden_dim, hidden_dim)

        # Recurrent connection
        self.W_rec = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Adaptive tau
        self.tau_net = nn.Linear(hidden_dim * 2, 1)

        # Norm
        self.norm = nn.LayerNorm(hidden_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.orthogonal_(self.W_rec.weight, gain=0.5)
        nn.init.xavier_uniform_(self.W_in.weight)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, hidden]
            h: [batch, hidden]
        Returns:
            new_h, tau
        """
        # Adaptive tau
        combined = torch.cat([x, h], dim=-1)
        tau_raw = torch.sigmoid(self.tau_net(combined))
        tau = self.tau_min + (self.tau_max - self.tau_min) * tau_raw

        # ODE dynamics
        f_xh = torch.tanh(self.W_in(x) + self.W_rec(h))
        dh = (-h + f_xh) / tau
        new_h = h + self.dt * dh
        new_h = self.norm(new_h)

        return new_h, tau


class LNNLite(nn.Module):
    """
    브라우저용 경량 LNN

    Architecture:
    - Input projection
    - 4 Liquid layers
    - Output projection

    ~5M params total
    """

    def __init__(
        self,
        input_dim: int = 384,      # Vision encoder output
        hidden_dim: int = 256,     # LNN hidden
        output_dim: int = 128,     # Action embedding
        num_layers: int = 4
    ):
        super().__init__()

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Liquid layers
        self.cells = nn.ModuleList([
            LiquidCellLite(hidden_dim)
            for _ in range(num_layers)
        ])

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)

        # Hidden states (for streaming)
        self.hidden_states: Optional[List[torch.Tensor]] = None

    def reset(self, batch_size: int = 1, device: str = "cpu"):
        """Reset hidden states"""
        self.hidden_states = [
            torch.zeros(batch_size, self.cells[0].hidden_dim, device=device)
            for _ in range(len(self.cells))
        ]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[float]]:
        """
        Args:
            x: [batch, input_dim] - single timestep
        Returns:
            output: [batch, output_dim]
            tau_values: list of tau for each layer
        """
        batch_size = x.shape[0]
        device = x.device

        # Initialize if needed
        if self.hidden_states is None:
            self.reset(batch_size, device)

        # Project input
        h = self.input_proj(x)

        # Liquid layers
        tau_values = []
        for i, cell in enumerate(self.cells):
            h, tau = cell(h, self.hidden_states[i])
            self.hidden_states[i] = h.detach()
            tau_values.append(tau.mean().item())

        # Output
        out = self.output_proj(h)

        return out, tau_values


class VisionLNNLite(nn.Module):
    """
    Vision + LNN 통합 모델 (브라우저용)

    MobileViT-XS (2.3M) + LNN (2.7M) = ~5M total
    """

    def __init__(
        self,
        num_actions: int = 6,
        use_tiny_vit: bool = True
    ):
        super().__init__()

        # Vision encoder (will load pretrained)
        if use_tiny_vit:
            self.vision_dim = 192
            self.vision = self._create_tiny_vision()
        else:
            self.vision_dim = 384
            self.vision = None  # Load externally

        # LNN core
        self.lnn = LNNLite(
            input_dim=self.vision_dim + 64,  # vision + task
            hidden_dim=256,
            output_dim=128,
            num_layers=4
        )

        # Task embedding
        self.task_embed = nn.Embedding(100, 64)

        # Action head
        self.action_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions)
        )

        # Coordinate head (for click x, y)
        self.coord_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Sigmoid()
        )

    def _create_tiny_vision(self) -> nn.Module:
        """Create minimal vision encoder"""
        return nn.Sequential(
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

    def forward(
        self,
        image: torch.Tensor,        # [batch, 3, H, W]
        task_id: torch.Tensor = None  # [batch]
    ) -> dict:
        """
        Forward pass

        Returns:
            action_logits: [batch, num_actions]
            coords: [batch, 2] (x, y in 0-1)
            tau_values: list of tau
        """
        batch_size = image.shape[0]

        # Vision encoding
        if self.vision is not None:
            vision_feat = self.vision(image)
        else:
            # External encoder
            vision_feat = image.view(batch_size, -1)[:, :self.vision_dim]

        # Task embedding
        if task_id is None:
            task_id = torch.zeros(batch_size, dtype=torch.long, device=image.device)
        task_feat = self.task_embed(task_id)

        # Combine
        combined = torch.cat([vision_feat, task_feat], dim=-1)

        # LNN processing
        lnn_out, tau_values = self.lnn(combined)

        # Heads
        action_logits = self.action_head(lnn_out)
        coords = self.coord_head(lnn_out)

        return {
            "action_logits": action_logits,
            "coords": coords,
            "tau_values": tau_values
        }

    def reset(self):
        """Reset LNN state"""
        self.lnn.hidden_states = None


def export_to_onnx(model: nn.Module, output_path: str = "lnn_lite.onnx"):
    """Export to ONNX for browser use"""
    model.eval()

    # Dummy inputs
    dummy_image = torch.randn(1, 3, 224, 224)
    dummy_task = torch.zeros(1, dtype=torch.long)

    # Export
    torch.onnx.export(
        model,
        (dummy_image, dummy_task),
        output_path,
        input_names=["image", "task_id"],
        output_names=["action_logits", "coords"],
        dynamic_axes={
            "image": {0: "batch"},
            "task_id": {0: "batch"}
        },
        opset_version=14
    )
    print(f"Exported to {output_path}")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def test():
    """Test the lite model"""
    print("=" * 50)
    print("LNN LITE TEST")
    print("=" * 50)

    # Test LNNLite
    print("\n[1] LNNLite")
    lnn = LNNLite(input_dim=384, hidden_dim=256, output_dim=128)
    print(f"    Params: {count_params(lnn):,} ({count_params(lnn)/1e6:.2f}M)")

    x = torch.randn(1, 384)
    for step in range(5):
        out, taus = lnn(x)
        print(f"    Step {step}: τ = {[f'{t:.2f}' for t in taus]}")

    # Test VisionLNNLite
    print("\n[2] VisionLNNLite (Full)")
    vlnn = VisionLNNLite(num_actions=6)
    print(f"    Params: {count_params(vlnn):,} ({count_params(vlnn)/1e6:.2f}M)")
    print(f"    Size (FP16): {count_params(vlnn) * 2 / 1e6:.1f} MB")

    img = torch.randn(1, 3, 224, 224)
    for step in range(3):
        result = vlnn(img)
        action_probs = F.softmax(result["action_logits"], dim=-1)
        coords = result["coords"]
        taus = result["tau_values"]
        print(f"    Step {step}: action={action_probs[0].argmax().item()}, "
              f"coords=({coords[0,0]:.2f}, {coords[0,1]:.2f}), "
              f"τ_avg={sum(taus)/len(taus):.2f}")

    # Memory size
    print("\n[3] 메모리 추정")
    fp32_mb = count_params(vlnn) * 4 / 1e6
    fp16_mb = count_params(vlnn) * 2 / 1e6
    int8_mb = count_params(vlnn) / 1e6
    print(f"    FP32: {fp32_mb:.1f} MB")
    print(f"    FP16: {fp16_mb:.1f} MB")
    print(f"    INT8: {int8_mb:.1f} MB")

    if fp16_mb < 20:
        print(f"\n    ✓ 브라우저 실행 가능! (WebGPU/WebNN)")
    else:
        print(f"\n    ❌ 브라우저에 너무 큼")

    print("\n" + "=" * 50)


if __name__ == "__main__":
    test()
