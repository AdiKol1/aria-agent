#!/usr/bin/env python3
"""Test finding dock icons with Claude - slow movement."""

import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_dock_slow():
    import pyautogui
    import anthropic
    from aria.vision import get_screen_capture
    from aria.config import ANTHROPIC_API_KEY

    screen_width, screen_height = pyautogui.size()
    print(f"Screen: {screen_width} x {screen_height}")

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    screen_capture = get_screen_capture()

    # Move mouse out of the way first
    print("Moving mouse to center...")
    pyautogui.moveTo(screen_width // 2, screen_height // 2, duration=0.5)
    time.sleep(1)

    # Capture screenshot
    print("Capturing screenshot...")
    result = screen_capture.capture_to_base64_with_size()
    if not result:
        print("Failed to capture")
        return

    image_b64, (w, h) = result
    print(f"Screenshot: {w} x {h}")

    icons_to_find = [
        ("Finder", "the Finder icon (blue and white smiley face) in the dock"),
        ("Trash", "the Trash icon (trash can) in the dock"),
        ("Chrome", "the Google Chrome icon in the dock"),
    ]

    for name, description in icons_to_find:
        print(f"\n{'='*50}")
        print(f"Finding: {name}")
        print(f"{'='*50}")

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_b64}},
                        {"type": "text", "text": f"""Image size: {w}x{h} pixels.

Find {description}.

Return ONLY JSON: {{"x": N, "y": N}}
x = pixels from LEFT edge of image
y = pixels from TOP edge of image
Give the CENTER of the icon."""}
                    ],
                }
            ],
        )

        text = response.content[0].text.strip()
        print(f"Claude says: {text[-100:]}")  # Last 100 chars

        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(text[start:end])
                x, y = data.get("x", 0), data.get("y", 0)

                # Scale to screen coords
                screen_x = int(x * screen_width / w)
                screen_y = int(y * screen_height / h)

                print(f"Screenshot coords: ({x}, {y})")
                print(f"Screen coords: ({screen_x}, {screen_y})")
                print(f"\n>>> WATCH THE MOUSE - Moving to {name} in 2 seconds...")
                time.sleep(2)

                pyautogui.moveTo(screen_x, screen_y, duration=1.0)
                print(f">>> Mouse should now be on {name}")
                print(">>> Is it correct? (waiting 4 seconds)")
                time.sleep(4)

        except Exception as e:
            print(f"Error: {e}")

    print("\n" + "="*50)
    print("TEST COMPLETE")
    print("="*50)


if __name__ == "__main__":
    test_dock_slow()
