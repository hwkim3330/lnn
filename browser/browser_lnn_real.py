#!/usr/bin/env python3
"""
REAL LNN Browser Agent

LFM2-VL + Liquid Neural Network 하이브리드
- LFM2-VL: 화면 이해 & 언어 추론
- LNN: 시간적 상태 유지 & 적응형 동역학

실제로 τ (time constant)가 변하고, 상태가 유지됩니다!
"""

import os
import sys
import time
import re
import json
from pathlib import Path
from typing import Tuple, Optional, Dict, List
from PIL import Image
import io

import torch
import torch.nn as nn

# Import our hybrid model
from lfm2_lnn_hybrid import LFM2LNNHybrid, LiquidCell


class LNNBrowserReasoner:
    """
    LNN-powered browser reasoning

    Maintains temporal state across browser interactions
    """

    def __init__(
        self,
        model_path: str = "/home/kim/models/lfm2-vl-450m",
        device: str = "cuda"
    ):
        print("=" * 60)
        print("LNN BROWSER REASONER")
        print("=" * 60)

        # Load hybrid model
        self.model = LFM2LNNHybrid(
            model_path=model_path,
            lnn_layers=[0, 4, 8, 12],
            device=device
        )

        # Additional LNN for action sequence memory
        self.action_memory = ActionMemoryLNN(
            action_dim=128,
            memory_dim=256,
            device=device
        )

        self.device = device
        self.action_history = []

    def reset(self):
        """Reset for new task"""
        self.model.reset_lnn_states()
        self.action_memory.reset()
        self.action_history = []

    def reason_about_screen(
        self,
        image: Image.Image,
        task: str,
        current_url: str = "",
        current_title: str = ""
    ) -> Dict:
        """
        Reason about what to do next

        Returns action dict with:
        - type: click/type/navigate/scroll/key/done
        - x, y: coordinates (if click)
        - text: text to type (if type)
        - etc.
        """
        # Build context-aware prompt
        history_str = ", ".join(self.action_history[-5:]) if self.action_history else "None"

        prompt = f"""You are a browser automation agent with memory.

TASK: {task}
URL: {current_url}
Title: {current_title}
Previous actions: {history_str}
Action count: {len(self.action_history)}

Look at the screenshot and decide the next action.

Available actions:
1. {{"type": "click", "x": <pixel_x>, "y": <pixel_y>, "reason": "why"}}
2. {{"type": "type", "text": "<text>", "reason": "why"}}
3. {{"type": "navigate", "url": "<url>", "reason": "why"}}
4. {{"type": "scroll", "direction": "down/up", "reason": "why"}}
5. {{"type": "key", "key": "Enter/Tab/Escape", "reason": "why"}}
6. {{"type": "done", "reason": "task completed"}}

Screen size: 1280x720 pixels.
Respond with ONLY a JSON object for the next action."""

        # Generate response with LNN-augmented model
        response = self.model.generate(
            image,
            prompt,
            max_new_tokens=256
        )

        print(f"    LNN Model: {response[:150]}...")

        # Parse action
        action = self._parse_action(response)

        # Update action memory LNN
        if action.get("type") != "unknown":
            self.action_memory.update(action)
            self.action_history.append(f"{action['type']}")

        return action

    def _parse_action(self, response: str) -> Dict:
        """Parse action from model response"""
        try:
            # Find JSON in response
            match = re.search(r'\{[^{}]+\}', response, re.DOTALL)
            if match:
                action = json.loads(match.group())
                return action
        except Exception as e:
            print(f"    Parse error: {e}")

        # Fallback parsing
        response_lower = response.lower()

        if "click" in response_lower:
            coords = re.findall(r'(\d+)', response)
            if len(coords) >= 2:
                return {
                    "type": "click",
                    "x": int(coords[0]),
                    "y": int(coords[1]),
                    "reason": "extracted from response"
                }

        elif "navigate" in response_lower or "go to" in response_lower:
            match = re.search(r'(https?://\S+|[\w.-]+\.(com|org|net|io|kr))', response)
            if match:
                return {
                    "type": "navigate",
                    "url": match.group(1),
                    "reason": "extracted URL"
                }

        elif "scroll" in response_lower:
            direction = "down" if "down" in response_lower else "up"
            return {"type": "scroll", "direction": direction, "reason": "scroll"}

        elif "type" in response_lower or "enter" in response_lower:
            match = re.search(r'["\']([^"\']+)["\']', response)
            if match:
                return {"type": "type", "text": match.group(1), "reason": "type text"}

        elif "done" in response_lower or "complete" in response_lower:
            return {"type": "done", "reason": "task completed"}

        return {"type": "unknown", "reason": "could not parse"}

    def get_lnn_state(self) -> Dict:
        """Get current LNN state info"""
        return {
            "action_count": len(self.action_history),
            "memory_state_norm": self.action_memory.get_state_norm(),
            "recent_actions": self.action_history[-5:]
        }


class ActionMemoryLNN(nn.Module):
    """
    Separate LNN for remembering action sequences

    Maintains state across browser interactions
    """

    def __init__(self, action_dim: int = 128, memory_dim: int = 256, device: str = "cuda"):
        super().__init__()

        self.device = device

        # Action encoder (float16 for consistency)
        self.action_encoder = nn.Sequential(
            nn.Linear(16, 64),  # Simple action encoding
            nn.ReLU(),
            nn.Linear(64, action_dim)
        ).to(device).half()

        # Liquid memory cell (float16)
        self.liquid = LiquidCell(
            input_dim=action_dim,
            hidden_dim=memory_dim,
            dt=0.2,
            tau_min=2.0,
            tau_max=8.0
        ).to(device).half()

        # State
        self.memory_state = None
        self.current_tau = 5.0

    def reset(self):
        """Reset memory state"""
        self.memory_state = None
        self.current_tau = 5.0

    def action_to_vector(self, action: Dict) -> torch.Tensor:
        """Convert action dict to vector"""
        vec = torch.zeros(16, device=self.device, dtype=torch.float16)

        action_type = action.get("type", "unknown")
        type_map = {
            "click": 0, "type": 1, "navigate": 2,
            "scroll": 3, "key": 4, "done": 5, "unknown": 6
        }
        vec[type_map.get(action_type, 6)] = 1.0

        # Position (normalized)
        if "x" in action:
            vec[7] = action["x"] / 1280
            vec[8] = action["y"] / 720

        # Direction
        if action.get("direction") == "down":
            vec[9] = 1.0
        elif action.get("direction") == "up":
            vec[10] = 1.0

        return vec

    @torch.no_grad()
    def update(self, action: Dict):
        """Update memory with new action"""
        # Encode action
        action_vec = self.action_to_vector(action)
        action_encoded = self.action_encoder(action_vec)  # [action_dim]

        # Add batch and seq dimensions
        action_seq = action_encoded.unsqueeze(0).unsqueeze(0)  # [1, 1, action_dim]

        # Update liquid memory
        output, new_state, tau = self.liquid(action_seq, self.memory_state)

        self.memory_state = new_state
        self.current_tau = tau.mean().item()

        return self.current_tau

    def get_state_norm(self) -> float:
        """Get norm of current memory state"""
        if self.memory_state is None:
            return 0.0
        return self.memory_state.norm().item()


class LNNBrowserAgent:
    """
    Full browser agent with LNN reasoning
    """

    def __init__(self, headless: bool = False):
        self.headless = headless
        self.screen_width = 1280
        self.screen_height = 720

        # Initialize reasoner
        self.reasoner = LNNBrowserReasoner()

        self.page = None

    def setup_browser(self):
        """Setup Playwright browser"""
        from playwright.sync_api import sync_playwright

        print("\nStarting browser...")
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=self.headless)
        self._context = self._browser.new_context(
            viewport={'width': self.screen_width, 'height': self.screen_height}
        )
        self.page = self._context.new_page()
        print(f"Browser ready ({self.screen_width}x{self.screen_height})")
        print("=" * 60)

    def cleanup(self):
        if hasattr(self, '_browser'):
            self._browser.close()
        if hasattr(self, '_pw'):
            self._pw.stop()

    def get_screenshot(self) -> Image.Image:
        """Get current screenshot"""
        screenshot_bytes = self.page.screenshot()
        return Image.open(io.BytesIO(screenshot_bytes)).convert("RGB")

    def execute_action(self, action: Dict) -> bool:
        """Execute an action"""
        action_type = action.get("type", "unknown")
        reason = action.get("reason", "")

        try:
            if action_type == "click":
                x = int(action.get("x", self.screen_width // 2))
                y = int(action.get("y", self.screen_height // 2))
                x = max(0, min(x, self.screen_width - 1))
                y = max(0, min(y, self.screen_height - 1))
                print(f"    🖱 Click ({x}, {y}) - {reason}")
                self.page.mouse.click(x, y)
                time.sleep(0.5)
                return True

            elif action_type == "type":
                text = action.get("text", "")
                print(f"    ⌨ Type: '{text}' - {reason}")
                self.page.keyboard.type(text)
                return True

            elif action_type == "navigate":
                url = action.get("url", "")
                if not url.startswith("http"):
                    url = "https://" + url
                print(f"    🌐 Navigate: {url} - {reason}")
                self.page.goto(url, wait_until="domcontentloaded", timeout=15000)
                time.sleep(1)
                return True

            elif action_type == "scroll":
                direction = action.get("direction", "down")
                delta = 300 if direction == "down" else -300
                icon = '↓' if direction == 'down' else '↑'
                print(f"    {icon} Scroll {direction} - {reason}")
                self.page.mouse.wheel(0, delta)
                return True

            elif action_type == "key":
                key = action.get("key", "Enter")
                print(f"    ⌨ Press: {key} - {reason}")
                self.page.keyboard.press(key)
                return True

            elif action_type == "done":
                print(f"    ✓ Done - {reason}")
                return True

            else:
                print(f"    ❓ Unknown: {action_type}")
                return False

        except Exception as e:
            print(f"    ✗ Error: {e}")
            return False

    def run_task(self, task: str, max_steps: int = 15) -> bool:
        """Run a task autonomously"""
        print(f"\n{'='*60}")
        print(f"📋 Task: {task}")
        print(f"{'='*60}")

        # Reset LNN states for new task
        self.reasoner.reset()

        for step in range(max_steps):
            print(f"\n  Step {step + 1}:")

            # Get screenshot
            image = self.get_screenshot()

            # Reason about what to do (with LNN!)
            action = self.reasoner.reason_about_screen(
                image,
                task,
                self.page.url,
                self.page.title()
            )

            # Show LNN state
            lnn_state = self.reasoner.get_lnn_state()
            print(f"    [LNN] Actions: {lnn_state['action_count']}, "
                  f"Memory norm: {lnn_state['memory_state_norm']:.2f}, "
                  f"τ: {self.reasoner.action_memory.current_tau:.2f}")

            # Execute
            success = self.execute_action(action)

            # Check if done
            if action.get("type") == "done":
                print(f"\n  ✓ Task completed in {step + 1} steps!")
                return True

            time.sleep(0.3)

        print(f"\n  ⚠ Max steps reached")
        return False


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--task", type=str, default=None)
    args = parser.parse_args()

    agent = LNNBrowserAgent(headless=args.headless)

    try:
        agent.setup_browser()

        if args.task:
            agent.run_task(args.task)
        else:
            # Demo tasks
            tasks = [
                "Go to google.com and search for 'liquid neural networks'",
            ]

            for task in tasks:
                agent.run_task(task)
                time.sleep(2)

    except KeyboardInterrupt:
        print("\nInterrupted!")
    finally:
        agent.cleanup()


if __name__ == "__main__":
    main()
