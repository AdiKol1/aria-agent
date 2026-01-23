#!/usr/bin/env python3
"""
Test Gemini's ability to accurately identify coordinates from screenshots.

This script:
1. Moves the mouse to a known position
2. Captures a screenshot
3. Asks Gemini to identify the mouse cursor position
4. Compares Gemini's answer to the actual position
"""

import os
import sys
import time
import json
import base64

# Add aria to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_coordinate_accuracy():
    import pyautogui
    from google import genai
    from aria.vision import get_screen_capture
    from aria.config import GOOGLE_API_KEY, SCREENSHOT_MAX_WIDTH, SCREENSHOT_QUALITY

    # Get screen dimensions
    screen_width, screen_height = pyautogui.size()
    print(f"Screen size: {screen_width} x {screen_height}")
    print(f"Screenshot max width: {SCREENSHOT_MAX_WIDTH}")
    print(f"Screenshot quality: {SCREENSHOT_QUALITY}")

    # Test positions - corners and center
    test_positions = [
        (screen_width // 2, screen_height // 2, "center"),
        (100, 100, "top-left"),
        (screen_width - 100, 100, "top-right"),
        (100, screen_height - 100, "bottom-left"),
        (screen_width - 100, screen_height - 100, "bottom-right"),
    ]

    # Initialize Gemini client
    client = genai.Client(api_key=GOOGLE_API_KEY)
    screen_capture = get_screen_capture()

    results = []

    for actual_x, actual_y, position_name in test_positions:
        print(f"\n--- Testing {position_name}: ({actual_x}, {actual_y}) ---")

        # Move mouse to known position
        pyautogui.moveTo(actual_x, actual_y)
        time.sleep(0.5)  # Wait for cursor to settle

        # Capture screenshot
        result = screen_capture.capture_to_base64_with_size()
        if not result:
            print("Failed to capture screenshot")
            continue

        image_b64, (screenshot_width, screenshot_height) = result
        print(f"Screenshot size: {screenshot_width} x {screenshot_height}")

        # Calculate expected coordinates in screenshot space
        scale_x = screenshot_width / screen_width
        scale_y = screenshot_height / screen_height
        expected_x = int(actual_x * scale_x)
        expected_y = int(actual_y * scale_y)
        print(f"Expected in screenshot coords: ({expected_x}, {expected_y})")

        # Ask Gemini to find the mouse cursor
        prompt = f"""Look at this screenshot ({screenshot_width} x {screenshot_height} pixels).

Find the mouse cursor in the image and return its CENTER coordinates.

Respond with ONLY a JSON object like this:
{{"x": 500, "y": 300, "found": true}}

If you cannot find the cursor, respond with:
{{"found": false, "reason": "why"}}

The x coordinate is pixels from the LEFT edge.
The y coordinate is pixels from the TOP edge.

Be precise - count the pixels carefully."""

        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash-exp",
                contents=[
                    {
                        "role": "user",
                        "parts": [
                            {"inline_data": {"mime_type": "image/jpeg", "data": image_b64}},
                            {"text": prompt}
                        ]
                    }
                ]
            )

            response_text = response.text.strip()
            print(f"Gemini response: {response_text}")

            # Parse JSON
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]

            result_json = json.loads(response_text)

            if result_json.get("found"):
                gemini_x = result_json["x"]
                gemini_y = result_json["y"]

                # Calculate error
                error_x = abs(gemini_x - expected_x)
                error_y = abs(gemini_y - expected_y)
                error_total = (error_x**2 + error_y**2) ** 0.5

                print(f"Gemini returned: ({gemini_x}, {gemini_y})")
                print(f"Expected: ({expected_x}, {expected_y})")
                print(f"Error: x={error_x}px, y={error_y}px, total={error_total:.1f}px")

                results.append({
                    "position": position_name,
                    "actual": (actual_x, actual_y),
                    "expected_screenshot": (expected_x, expected_y),
                    "gemini": (gemini_x, gemini_y),
                    "error_x": error_x,
                    "error_y": error_y,
                    "error_total": error_total
                })
            else:
                print(f"Gemini could not find cursor: {result_json.get('reason', 'unknown')}")
                results.append({
                    "position": position_name,
                    "actual": (actual_x, actual_y),
                    "found": False,
                    "reason": result_json.get("reason", "unknown")
                })

        except Exception as e:
            print(f"Error: {e}")
            results.append({
                "position": position_name,
                "error": str(e)
            })

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    successful = [r for r in results if "error_total" in r]
    if successful:
        avg_error = sum(r["error_total"] for r in successful) / len(successful)
        max_error = max(r["error_total"] for r in successful)
        print(f"Successful tests: {len(successful)}/{len(results)}")
        print(f"Average error: {avg_error:.1f} pixels")
        print(f"Max error: {max_error:.1f} pixels")

        # Interpretation
        if avg_error < 20:
            print("\n✓ GOOD: Gemini can identify coordinates with reasonable accuracy")
        elif avg_error < 50:
            print("\n⚠ MODERATE: Gemini has some accuracy but may miss small targets")
        else:
            print("\n✗ POOR: Gemini cannot reliably identify pixel coordinates")
    else:
        print("No successful tests")

    return results


if __name__ == "__main__":
    test_coordinate_accuracy()
