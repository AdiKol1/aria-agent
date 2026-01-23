#!/usr/bin/env python3
"""Test coordinate consistency - find same element multiple times."""

import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_consistency():
    import anthropic
    from aria.vision import get_screen_capture
    from aria.config import ANTHROPIC_API_KEY

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    screen_capture = get_screen_capture()

    # Capture one screenshot
    result = screen_capture.capture_to_base64_with_size()
    if not result:
        print("Failed to capture screenshot")
        return

    image_b64, (w, h) = result
    print(f"Screenshot: {w} x {h}")

    # Find Apple menu 5 times with same screenshot
    element = "the Apple logo in the top-left corner of the menu bar"
    results = []

    for i in range(5):
        print(f"\nAttempt {i+1}:")

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": "image/jpeg", "data": image_b64},
                        },
                        {
                            "type": "text",
                            "text": f"""Image size: {w}x{h} pixels. Find: "{element}"
Return ONLY JSON: {{"x": N, "y": N}}
x=pixels from left, y=pixels from top. Give the CENTER of the element."""
                        },
                    ],
                }
            ],
        )

        text = response.content[0].text.strip()
        print(f"  Response: {text}")

        # Extract coordinates
        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(text[start:end])
                x, y = data.get("x", 0), data.get("y", 0)
                results.append((x, y))
                print(f"  Coords: ({x}, {y})")
        except:
            print(f"  Failed to parse")

        time.sleep(1)

    # Analyze consistency
    if results:
        print(f"\n{'='*50}")
        print("CONSISTENCY ANALYSIS")
        print(f"{'='*50}")
        print(f"Results: {results}")

        xs = [r[0] for r in results]
        ys = [r[1] for r in results]

        x_range = max(xs) - min(xs)
        y_range = max(ys) - min(ys)

        print(f"X range: {min(xs)} - {max(xs)} (spread: {x_range}px)")
        print(f"Y range: {min(ys)} - {max(ys)} (spread: {y_range}px)")

        if x_range < 50 and y_range < 50:
            print("\n✓ CONSISTENT: Coordinates are stable")
        else:
            print("\n✗ INCONSISTENT: Large variation in coordinates")


if __name__ == "__main__":
    test_consistency()
