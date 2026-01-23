#!/usr/bin/env python3
"""
Test Claude's ability to find UI elements and return coordinates.
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_claude_find_element():
    import pyautogui
    import anthropic
    from aria.vision import get_screen_capture
    from aria.config import ANTHROPIC_API_KEY

    screen_width, screen_height = pyautogui.size()
    print(f"Screen size: {screen_width} x {screen_height}")

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    screen_capture = get_screen_capture()

    # Capture screenshot
    result = screen_capture.capture_to_base64_with_size()
    if not result:
        print("Failed to capture screenshot")
        return

    image_b64, (screenshot_width, screenshot_height) = result
    print(f"Screenshot size: {screenshot_width} x {screenshot_height}")

    # Test finding different elements
    test_elements = [
        "the Apple menu (Apple logo) in the top-left corner of the menu bar",
        "the Finder icon in the dock",
        "the trash icon in the dock",
        "the Safari icon in the dock (if visible)",
        "the Terminal icon in the dock (if visible)",
    ]

    for element in test_elements:
        print(f"\n--- Finding: {element} ---")

        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": image_b64,
                                },
                            },
                            {
                                "type": "text",
                                "text": f"""Screenshot dimensions: {screenshot_width} x {screenshot_height} pixels

Find this UI element: "{element}"

Return the CENTER coordinates of the element as JSON:
{{"found": true, "x": 500, "y": 300, "element": "description"}}

Or if not found:
{{"found": false, "reason": "why"}}

IMPORTANT:
- x = pixels from LEFT edge (0 = left edge, {screenshot_width} = right edge)
- y = pixels from TOP edge (0 = top edge, {screenshot_height} = bottom edge)
- Return ONLY the JSON, no other text"""
                            },
                        ],
                    }
                ],
            )

            response_text = response.content[0].text.strip()

            # Extract JSON from response
            if "```" in response_text:
                parts = response_text.split("```")
                for part in parts:
                    if part.strip().startswith("json"):
                        response_text = part.strip()[4:]
                        break
                    elif part.strip().startswith("{"):
                        response_text = part.strip()
                        break

            # Find JSON in response
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            if start >= 0 and end > start:
                json_str = response_text[start:end]
                result_json = json.loads(json_str)

                if result_json.get("found"):
                    x, y = result_json["x"], result_json["y"]
                    print(f"  Found at: ({x}, {y}) in screenshot coords")

                    # Convert to screen coords
                    scale_x = screen_width / screenshot_width
                    scale_y = screen_height / screenshot_height
                    screen_x = int(x * scale_x)
                    screen_y = int(y * scale_y)
                    print(f"  Screen coords: ({screen_x}, {screen_y})")

                    # Ask user to verify
                    print(f"  Moving mouse there to verify...")
                    pyautogui.moveTo(screen_x, screen_y)
                else:
                    print(f"  Not found: {result_json.get('reason', 'unknown')}")
            else:
                print(f"  Could not parse response: {response_text[:200]}")

        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()

        import time
        time.sleep(2)  # Pause so user can see where mouse moved


if __name__ == "__main__":
    test_claude_find_element()
