#!/usr/bin/env python3
"""
Browser Agent using LFM2-VL

Uses the full LFM2-VL model for:
1. Understanding screenshots
2. Reasoning about tasks
3. Generating actions

This is like a mini Computer Use agent.
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
from transformers import AutoModelForImageTextToText, AutoProcessor


class LFM2BrowserAgent:
    """Browser agent powered by LFM2-VL"""

    def __init__(
        self,
        model_path: str = "/home/kim/models/lfm2-vl-450m",
        headless: bool = False
    ):
        self.headless = headless
        self.screen_width = 1280
        self.screen_height = 720

        print("=" * 60)
        print("LFM2-VL BROWSER AGENT")
        print("=" * 60)

        # Load model
        print("\n[1/2] Loading LFM2-VL...")
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="cuda",
            trust_remote_code=True,
            local_files_only=True
        )
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True
        )
        self.model.eval()
        print(f"    Model loaded on CUDA")

        self.page = None
        self.action_history = []

    def setup_browser(self):
        """Setup browser"""
        from playwright.sync_api import sync_playwright

        print("\n[2/2] Starting browser...")
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=self.headless)
        self._context = self._browser.new_context(
            viewport={'width': self.screen_width, 'height': self.screen_height}
        )
        self.page = self._context.new_page()
        print(f"    Browser ready ({self.screen_width}x{self.screen_height})")
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

    def ask_model(self, image: Image.Image, prompt: str) -> str:
        """Ask LFM2-VL a question about the image"""
        # Build conversation
        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]
        }]

        # Apply chat template
        text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True
        )

        # Process inputs
        inputs = self.processor(
            images=[image],
            text=text,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=self.processor.tokenizer.pad_token_id
            )

        # Decode - only new tokens
        response = self.processor.decode(
            output_ids[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        return response.strip()

    def decide_action(self, task: str) -> Dict:
        """
        Look at screen and decide what to do

        Returns action dict like:
        {"type": "click", "x": 640, "y": 360, "reason": "clicking search box"}
        {"type": "type", "text": "hello", "reason": "typing search query"}
        {"type": "navigate", "url": "google.com", "reason": "going to google"}
        {"type": "scroll", "direction": "down", "reason": "scrolling to see more"}
        {"type": "key", "key": "Enter", "reason": "submitting form"}
        {"type": "done", "reason": "task completed"}
        """
        image = self.get_screenshot()

        # Build prompt
        prompt = f"""You are a browser automation agent.

Current task: {task}
Current URL: {self.page.url}
Current title: {self.page.title()}
Previous actions: {self.action_history[-3:] if self.action_history else "None"}

Look at the screenshot and decide what action to take next.

Available actions:
1. click(x, y) - click at pixel coordinates
2. type(text) - type text
3. navigate(url) - go to a URL
4. scroll(direction) - scroll up or down
5. key(key_name) - press a key like Enter, Tab, Escape
6. done() - task is complete

Respond with a JSON object:
{{"type": "action_type", "x": 640, "y": 360, "text": "...", "url": "...", "direction": "...", "key": "...", "reason": "why this action"}}

What is the next action?"""

        response = self.ask_model(image, prompt)
        print(f"    Model: {response[:150]}...")

        # Parse JSON from response
        try:
            # Find JSON in response
            match = re.search(r'\{[^}]+\}', response, re.DOTALL)
            if match:
                action = json.loads(match.group())
                return action
        except Exception as e:
            print(f"    Parse error: {e}")

        # Fallback: try to understand the response
        response_lower = response.lower()
        if "click" in response_lower:
            # Try to extract coordinates
            coords = re.findall(r'(\d+)', response)
            if len(coords) >= 2:
                return {"type": "click", "x": int(coords[0]), "y": int(coords[1]), "reason": "extracted from response"}
        elif "type" in response_lower or "enter" in response_lower:
            match = re.search(r'["\']([^"\']+)["\']', response)
            if match:
                return {"type": "type", "text": match.group(1), "reason": "extracted text"}
        elif "scroll" in response_lower:
            direction = "down" if "down" in response_lower else "up"
            return {"type": "scroll", "direction": direction, "reason": "scroll requested"}
        elif "navigate" in response_lower or "go to" in response_lower:
            match = re.search(r'(https?://\S+|[\w.-]+\.(com|org|net|io))', response)
            if match:
                return {"type": "navigate", "url": match.group(1), "reason": "extracted URL"}

        return {"type": "unknown", "reason": "could not parse response"}

    def execute_action(self, action: Dict) -> bool:
        """Execute an action"""
        action_type = action.get("type", "unknown")
        reason = action.get("reason", "")

        try:
            if action_type == "click":
                x = int(action.get("x", self.screen_width // 2))
                y = int(action.get("y", self.screen_height // 2))
                # Clamp to screen
                x = max(0, min(x, self.screen_width - 1))
                y = max(0, min(y, self.screen_height - 1))
                print(f"    🖱 Click ({x}, {y}) - {reason}")
                self.page.mouse.click(x, y)
                time.sleep(0.5)
                self.action_history.append(f"click({x},{y})")
                return True

            elif action_type == "type":
                text = action.get("text", "")
                print(f"    ⌨ Type: '{text}' - {reason}")
                self.page.keyboard.type(text)
                self.action_history.append(f"type('{text}')")
                return True

            elif action_type == "navigate":
                url = action.get("url", "")
                if not url.startswith("http"):
                    url = "https://" + url
                print(f"    🌐 Navigate: {url} - {reason}")
                self.page.goto(url, wait_until="domcontentloaded", timeout=15000)
                time.sleep(1)
                self.action_history.append(f"navigate({url})")
                return True

            elif action_type == "scroll":
                direction = action.get("direction", "down")
                delta = 300 if direction == "down" else -300
                print(f"    {'↓' if direction == 'down' else '↑'} Scroll {direction} - {reason}")
                self.page.mouse.wheel(0, delta)
                self.action_history.append(f"scroll({direction})")
                return True

            elif action_type == "key":
                key = action.get("key", "Enter")
                print(f"    ⌨ Press: {key} - {reason}")
                self.page.keyboard.press(key)
                self.action_history.append(f"key({key})")
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

        self.action_history = []

        for step in range(max_steps):
            print(f"\n  Step {step + 1}:")

            # Get action from model
            action = self.decide_action(task)

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

    agent = LFM2BrowserAgent(headless=args.headless)

    try:
        agent.setup_browser()

        if args.task:
            agent.run_task(args.task)
        else:
            # Demo tasks
            tasks = [
                "Go to google.com and search for 'liquid neural networks'",
                "Go to github.com",
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
