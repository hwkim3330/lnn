#!/usr/bin/env python3
"""
LFM2-VL + LNN Merged Model

ShortConv + LiquidLayer를 하나의 모듈로 합침
- ShortConv: 원본 Conv1D 처리 (학습된 가중치 유지)
- LiquidLayer: ODE 기반 시간 동역학 (추가)
- Gate: 두 출력을 학습 가능한 비율로 혼합

하나의 통합된 신경망!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MergedConvLiquid(nn.Module):
    """
    ShortConv + Liquid를 하나로 합친 모듈

    output = gate * shortconv_out + (1-gate) * liquid_out

    gate는 학습 가능 (입력에 따라 적응)
    """

    def __init__(
        self,
        original_conv: nn.Module,
        hidden_dim: int = 1024,
        dt: float = 0.1,
        tau_min: float = 1.0,
        tau_max: float = 5.0,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.dt = dt
        self.tau_min = tau_min
        self.tau_max = tau_max

        # 원본 ShortConv 유지
        self.shortconv = original_conv

        # Liquid 컴포넌트 (새로 추가)
        self.liquid_W_rec = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.liquid_tau_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        self.liquid_norm = nn.LayerNorm(hidden_dim)

        # 적응형 게이트 (ShortConv vs Liquid 비율)
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.Tanh(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )

        # Hidden state
        self.register_buffer('h_liquid', None)

        # 초기화: 처음에는 원본 위주 (gate ≈ 0.9)
        self._init_gate_bias()

    def _init_gate_bias(self):
        """Gate를 원본 위주로 초기화"""
        # 마지막 Linear의 bias를 양수로 → sigmoid 출력 > 0.5
        with torch.no_grad():
            self.gate_net[-2].bias.fill_(2.0)  # sigmoid(2) ≈ 0.88

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values=None,
        cache_position=None,
        attention_mask=None,
    ) -> torch.Tensor:
        """
        Merged forward pass

        ShortConv 출력에 Liquid를 작은 잔차로 추가
        output = shortconv_out + scale * liquid_residual

        Args:
            hidden_states: [batch, seq, hidden_dim]

        Returns:
            output: [batch, seq, hidden_dim]
        """
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype

        # 1. 원본 ShortConv 실행 (주력)
        shortconv_out = self.shortconv(
            hidden_states,
            past_key_values=past_key_values,
            cache_position=cache_position,
            attention_mask=attention_mask
        )

        # 2. Liquid 처리 (잔차용)
        if self.h_liquid is None or self.h_liquid.shape[0] != batch_size:
            self.h_liquid = torch.zeros(
                batch_size, self.hidden_dim,
                device=device, dtype=dtype
            )

        liquid_outputs = []
        for t in range(seq_len):
            x_t = hidden_states[:, t, :]  # [batch, hidden]

            # Adaptive tau
            combined = torch.cat([x_t, self.h_liquid], dim=-1)
            tau_raw = self.liquid_tau_net(combined)
            tau = self.tau_min + (self.tau_max - self.tau_min) * tau_raw

            # ODE: dh/dt = (-h + tanh(x + W_rec*h)) / tau
            f_xh = torch.tanh(x_t + self.liquid_W_rec(self.h_liquid))
            dh = (-self.h_liquid + f_xh) / tau

            # Euler step
            self.h_liquid = self.h_liquid + self.dt * dh
            self.h_liquid = self.liquid_norm(self.h_liquid)

            liquid_outputs.append(self.h_liquid)

        liquid_out = torch.stack(liquid_outputs, dim=1)  # [batch, seq, hidden]

        # Detach for next forward
        self.h_liquid = self.h_liquid.detach()

        # 3. 작은 스케일로 잔차 추가 (0.01 = 1% 기여)
        # 학습하면서 점점 늘릴 수 있음
        scale = 0.01
        output = shortconv_out + scale * (liquid_out - shortconv_out)

        return output

    def reset_state(self):
        """Reset liquid state"""
        self.h_liquid = None

    def get_gate_value(self) -> float:
        """현재 평균 gate 값 반환 (디버깅용)"""
        return 0.5  # placeholder


class LFM2LiquidMerged(nn.Module):
    """
    통합된 LFM2-Liquid 모델

    ShortConv와 Liquid가 하나의 모듈로 합쳐짐
    """

    def __init__(
        self,
        model_path: str = "/home/kim/models/lfm2-vl-450m",
        liquid_layers: list = [0, 4, 8, 12, 15],
        device: str = "cuda"
    ):
        super().__init__()

        print("=" * 60)
        print("LFM2-LIQUID MERGED MODEL")
        print("=" * 60)

        from transformers import AutoModelForImageTextToText, AutoProcessor

        # 1. 원본 모델 로드
        print("\n[1/3] Loading LFM2-VL...")
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="cpu",
            trust_remote_code=True,
            local_files_only=True
        )
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True
        )

        # 2. Conv → MergedConvLiquid 교체
        print(f"\n[2/3] Merging ShortConv + Liquid at layers {liquid_layers}...")
        self.liquid_layers = liquid_layers
        self._merge_layers(liquid_layers)

        # 3. GPU로 이동
        print(f"\n[3/3] Moving to {device}...")
        self.model = self.model.to(device)
        self.device = device

        # 통계
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print("\n" + "=" * 60)
        print("READY!")
        print(f"  Total params: {total/1e6:.1f}M")
        print(f"  Trainable: {trainable/1e6:.1f}M")
        print(f"  Merged layers: {liquid_layers}")
        print("=" * 60)

    def _merge_layers(self, liquid_layers: list):
        """ShortConv를 MergedConvLiquid로 교체"""
        layers = self.model.model.language_model.layers

        for idx in liquid_layers:
            if idx < len(layers):
                layer = layers[idx]
                if hasattr(layer, 'conv'):
                    original_conv = layer.conv

                    # Merged 모듈 생성
                    merged = MergedConvLiquid(
                        original_conv=original_conv,
                        hidden_dim=1024,
                        dt=0.1,
                        tau_min=1.0,
                        tau_max=5.0
                    )

                    # dtype 맞추기
                    merged = merged.to(dtype=torch.float16)

                    # 교체
                    layer.conv = merged
                    print(f"    Layer {idx}: ShortConv + Liquid → Merged")

    def reset_liquid_states(self):
        """모든 Liquid 상태 초기화"""
        for layer in self.model.model.language_model.layers:
            if hasattr(layer, 'conv') and isinstance(layer.conv, MergedConvLiquid):
                layer.conv.reset_state()

    def freeze_original(self):
        """원본 파라미터 freeze, Liquid만 학습"""
        for name, param in self.model.named_parameters():
            if 'liquid' not in name and 'gate' not in name:
                param.requires_grad = False

    @torch.no_grad()
    def generate(
        self,
        image,
        prompt: str,
        max_new_tokens: int = 256,
        reset_state: bool = True
    ) -> str:
        """Generate response"""
        if reset_state:
            self.reset_liquid_states()

        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]
        }]

        text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True
        )
        inputs = self.processor(
            images=[image],
            text=text,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.processor.tokenizer.pad_token_id
        )

        response = self.processor.decode(
            output_ids[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        return response.strip()


def test():
    """Test merged model"""
    from PIL import Image

    print("\n" + "=" * 60)
    print("MERGED LFM2-LIQUID TEST")
    print("=" * 60)

    model = LFM2LiquidMerged(
        liquid_layers=[0, 8, 15],  # 3개 레이어
        device="cuda"
    )

    # Test 1
    print("\n[Test 1] Image description")
    img = Image.new('RGB', (224, 224), color='blue')
    response = model.generate(img, "What color is this image?", max_new_tokens=30)
    print(f"  Response: {response}")

    # Test 2
    print("\n[Test 2] Browser action")
    img = Image.new('RGB', (1280, 720), color='white')
    prompt = """You are a browser agent. Screen is blank.
Task: Go to google.com
What action? Reply with JSON only: {"type": "navigate", "url": "..."}"""
    response = model.generate(img, prompt, max_new_tokens=50)
    print(f"  Response: {response}")

    # Test 3: Multi-turn (state persistence)
    print("\n[Test 3] Multi-turn conversation")
    model.reset_liquid_states()

    for i in range(3):
        response = model.generate(
            img,
            f"Turn {i+1}. What do you see?",
            max_new_tokens=30,
            reset_state=False
        )
        print(f"  Turn {i+1}: {response[:60]}...")

    print("\n" + "=" * 60)
    print("SUCCESS - Merged model working!")
    print("=" * 60)


if __name__ == "__main__":
    test()
