#!/usr/bin/env python3
"""
zoomclick - Iterative zoom-and-click tool for AI-assisted UI automation

Workflow:
1. Start: zoomclick --start → full screenshot with quadrant overlay
2. Zoom:  zoomclick --zoom <quadrant> → zoom into selected region
3. Save:  zoomclick --save <name> → save current view as clickable template  
4. Click: zoomclick --click <name> → find saved template and click it
5. List:  zoomclick --list → list saved templates

Quadrants: top-left, top-right, bottom-left, bottom-right, center

The AI iteratively zooms until the target is big and centered, then saves
the template. Later, the template can be clicked without re-zooming.

Note: Guide line overlays are only added to output images for navigation.
They never appear on the actual screen or in saved templates. Each operation
takes a fresh screenshot directly from the display.

Window/Screen Targeting:
  zoomclick --start --window "Chrome"      # Start on specific window by title
  zoomclick --start --window-class "firefox" # Start on window by class
  zoomclick --start --window-id 12345      # Start on specific window ID
  zoomclick --start --screen 1             # Start on specific screen (multi-monitor)
  zoomclick --list-windows                 # List all windows with IDs

Usage Examples:
  zoomclick --start                    # Take screenshot, show with quadrant guides
  zoomclick --start -w "Firefox"       # Start zooming on Firefox window
  zoomclick --zoom top-left            # Zoom into top-left quadrant
  zoomclick --zoom center              # Zoom into center region
  zoomclick --zoom center              # Keep zooming until element is big
  zoomclick --save "submit_button"     # Save current zoomed region as template
  zoomclick --click "submit_button"    # Find and click the saved template
  zoomclick --list                     # List all saved templates
  zoomclick --reset                    # Reset zoom state (start fresh)
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Tuple

# Set display before importing pyautogui
os.environ.setdefault('DISPLAY', ':99')

# Suppress mouseinfo tkinter warning  
sys.modules['mouseinfo'] = type(sys)('mouseinfo')

import pyautogui

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.1

# Import helpers
from helpers import (
    WORK_DIR, TEMPLATES_DIR, STATE_FILE,
    find_window_by_name, find_window_by_class, get_window_geometry, list_windows,
    take_screenshot, take_screenshot_window, take_screenshot_screen,
    crop_image, add_quadrant_overlay, get_screen_size
)

@dataclass
class ViewportState:
    """Tracks the current viewport region on screen."""
    x: int = 0           # Top-left X of current viewport in screen coords
    y: int = 0           # Top-left Y of current viewport in screen coords
    width: int = 0       # Viewport width
    height: int = 0      # Viewport height
    screen_width: int = 0
    screen_height: int = 0
    zoom_level: int = 0  # How many times we've zoomed
    window_id: int = 0   # Window ID if targeting a window (0 = full screen)
    window_offset_x: int = 0  # Window X position on screen
    window_offset_y: int = 0  # Window Y position on screen
    
    def to_dict(self):
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d):
        return cls(**d)
    
    def save(self):
        with open(STATE_FILE, 'w') as f:
            json.dump(self.to_dict(), f)
    
    @classmethod
    def load(cls) -> Optional['ViewportState']:
        if STATE_FILE.exists():
            with open(STATE_FILE) as f:
                return cls.from_dict(json.load(f))
        return None


def get_quadrant_bounds(state: ViewportState, quadrant: str) -> Tuple[int, int, int, int]:
    """
    Get the bounds (x, y, width, height) for a quadrant within current viewport.
    Returns coordinates relative to the SCREEN (not the cropped image).
    """
    vw, vh = state.width, state.height
    vx, vy = state.x, state.y
    
    # Quadrant sizes (half of viewport)
    half_w = vw // 2
    half_h = vh // 2
    
    # Center region (middle 50%)
    quarter_w = vw // 4
    quarter_h = vh // 4
    
    if quadrant == "top-left":
        return (vx, vy, half_w, half_h)
    elif quadrant == "top-right":
        return (vx + half_w, vy, half_w, half_h)
    elif quadrant == "bottom-left":
        return (vx, vy + half_h, half_w, half_h)
    elif quadrant == "bottom-right":
        return (vx + half_w, vy + half_h, half_w, half_h)
    elif quadrant == "center":
        return (vx + quarter_w, vy + quarter_h, half_w, half_h)
    else:
        raise ValueError(f"Unknown quadrant: {quadrant}. Use: top-left, top-right, bottom-left, bottom-right, center")

def start_session(window_id: int = None, screen_num: int = None) -> dict:
    """Start a new zoom session with full screenshot or window/screen capture."""
    screen_w, screen_h = get_screen_size()
    
    window_offset_x = 0
    window_offset_y = 0
    capture_width = screen_w
    capture_height = screen_h
    
    if window_id:
        # Capture specific window
        screenshot_path = take_screenshot_window(window_id, "window")
        geo = get_window_geometry(window_id)
        window_offset_x = geo.get("X", 0)
        window_offset_y = geo.get("Y", 0)
        capture_width = geo.get("WIDTH", screen_w)
        capture_height = geo.get("HEIGHT", screen_h)
    elif screen_num is not None:
        # Capture specific screen
        screenshot_path = take_screenshot_screen(screen_num, "screen")
        # Get screen geometry for offset tracking
        import re
        result = subprocess.run(['xrandr'], capture_output=True, text=True)
        screens = []
        for line in result.stdout.split('\n'):
            if ' connected' in line:
                match = re.search(r'(\d+)x(\d+)\+(\d+)\+(\d+)', line)
                if match:
                    screens.append({
                        'width': int(match.group(1)),
                        'height': int(match.group(2)),
                        'x': int(match.group(3)),
                        'y': int(match.group(4))
                    })
        if screen_num < len(screens):
            s = screens[screen_num]
            window_offset_x = s['x']
            window_offset_y = s['y']
            capture_width = s['width']
            capture_height = s['height']
    else:
        # Full screen capture
        screenshot_path = take_screenshot("full")
    
    # Initialize viewport state
    state = ViewportState(
        x=0, y=0,
        width=capture_width, height=capture_height,
        screen_width=screen_w, screen_height=screen_h,
        zoom_level=0,
        window_id=window_id or 0,
        window_offset_x=window_offset_x,
        window_offset_y=window_offset_y
    )
    state.save()
    
    # Create overlay version
    overlay_path = WORK_DIR / f"overlay_{int(time.time())}.png"
    add_quadrant_overlay(screenshot_path, overlay_path, capture_width, capture_height)
    
    result = {
        "success": True,
        "action": "start",
        "screenshot": str(overlay_path),
        "viewport": state.to_dict(),
        "instructions": """
Analyze the screenshot. The image shows:
- Red lines dividing into 4 quadrants (top-left, top-right, bottom-left, bottom-right)
- Green box showing the CENTER region

Choose which quadrant contains your target element, then run:
  zoomclick --zoom <quadrant>

Quadrants: top-left, top-right, bottom-left, bottom-right, center

Keep zooming until your target is BIG and in the CENTER of the image.
Then save it with: zoomclick --save "button_name"
""".strip()
    }
    
    if window_id:
        result["window_id"] = window_id
        result["window_position"] = {"x": window_offset_x, "y": window_offset_y}
    
    return result

def zoom_to_quadrant(quadrant: str) -> dict:
    """Zoom into a quadrant of the current viewport."""
    state = ViewportState.load()
    if not state:
        return {"success": False, "error": "No active session. Run: zoomclick --start"}
    
    # Get new viewport bounds
    new_x, new_y, new_w, new_h = get_quadrant_bounds(state, quadrant)
    
    # Take fresh screenshot (window or full screen based on session)
    if state.window_id:
        screenshot_path = take_screenshot_window(state.window_id, "window")
    else:
        screenshot_path = take_screenshot("full")
    
    # Crop to new viewport
    cropped_path = WORK_DIR / f"zoom_{state.zoom_level + 1}_{int(time.time())}.png"
    crop_image(screenshot_path, new_x, new_y, new_w, new_h, cropped_path)
    
    # Update state (preserve window tracking)
    state.x = new_x
    state.y = new_y
    state.width = new_w
    state.height = new_h
    state.zoom_level += 1
    state.save()
    
    # Add overlay to cropped image
    overlay_path = WORK_DIR / f"overlay_{state.zoom_level}_{int(time.time())}.png"
    add_quadrant_overlay(cropped_path, overlay_path, new_w, new_h)
    
    # Calculate actual screen coordinates (accounting for window offset)
    screen_center_x = new_x + new_w // 2 + state.window_offset_x
    screen_center_y = new_y + new_h // 2 + state.window_offset_y
    
    result = {
        "success": True,
        "action": "zoom",
        "quadrant": quadrant,
        "screenshot": str(overlay_path),
        "viewport": state.to_dict(),
        "screen_coords": {
            "center_x": screen_center_x,
            "center_y": screen_center_y,
            "description": "If you clicked now, this would be the screen coordinate"
        },
        "instructions": f"""
Zoomed into {quadrant}. Now at zoom level {state.zoom_level}.
Viewport: {new_w}x{new_h} at position ({new_x}, {new_y})

If your target is BIG and CENTERED, save it:
  zoomclick --save "button_name"

If you need to zoom more, pick another quadrant:
  zoomclick --zoom <quadrant>

To click the current center immediately (without saving):
  zoomclick --click-center
""".strip()
    }
    
    if state.window_id:
        result["window_id"] = state.window_id
    
    return result

def save_template(name: str) -> dict:
    """Save current viewport as a reusable template."""
    state = ViewportState.load()
    if not state:
        return {"success": False, "error": "No active session. Run: zoomclick --start"}
    
    # Take fresh screenshot (window or full screen based on session)
    if state.window_id:
        screenshot_path = take_screenshot_window(state.window_id, "window")
    else:
        screenshot_path = take_screenshot("full")
    
    # Add timestamp to avoid name collisions
    timestamp = int(time.time())
    full_name = f"{name}_{timestamp}"
    
    template_path = TEMPLATES_DIR / f"{full_name}.png"
    meta_path = TEMPLATES_DIR / f"{full_name}.json"
    
    crop_image(screenshot_path, state.x, state.y, state.width, state.height, template_path)
    
    # Save metadata - viewport center + screen-absolute coordinates
    viewport_center_x = state.x + state.width // 2
    viewport_center_y = state.y + state.height // 2
    screen_center_x = viewport_center_x + state.window_offset_x
    screen_center_y = viewport_center_y + state.window_offset_y
    
    meta = {
        "name": full_name,
        "base_name": name,
        "viewport_x": viewport_center_x,
        "viewport_y": viewport_center_y,
        "center_x": screen_center_x,  # Screen-absolute for clicking
        "center_y": screen_center_y,
        "window_id": state.window_id,
        "window_offset_x": state.window_offset_x,
        "window_offset_y": state.window_offset_y,
        "viewport": state.to_dict(),
        "created": timestamp,
        "note": "Template saved for future clicking. Use: zoomclick --click " + full_name
    }
    
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    
    result = {
        "success": True,
        "action": "save",
        "name": full_name,
        "base_name": name,
        "template_path": str(template_path),
        "viewport_coords": {"x": viewport_center_x, "y": viewport_center_y},
        "screen_coords": {"x": screen_center_x, "y": screen_center_y},
        "instructions": f"""
Template "{full_name}" saved!
- Image: {template_path}
- Screen coordinates: ({screen_center_x}, {screen_center_y})

To click this element anytime, run:
  zoomclick --click "{full_name}"

The tool will find the template on screen and click its center.
""".strip()
    }
    
    if state.window_id:
        result["window_id"] = state.window_id
    
    return result

def find_template_on_screen(template_path: Path, screenshot_path: Path, min_confidence: float = 0.5) -> Optional[Tuple[int, int, float]]:
    """Find template on screen using OpenCV. Returns (center_x, center_y, confidence) or None."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None
    
    screen = cv2.imread(str(screenshot_path))
    template = cv2.imread(str(template_path))
    
    if screen is None or template is None:
        return None
    
    # Adaptive confidence matching
    confidence = 1.0
    while confidence >= min_confidence:
        result = cv2.matchTemplate(screen, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        if max_val >= confidence:
            h, w = template.shape[:2]
            center_x = max_loc[0] + w // 2
            center_y = max_loc[1] + h // 2
            return (center_x, center_y, max_val)
        
        confidence -= 0.1
    
    return None

def click_template(name: str, no_click: bool = False) -> dict:
    """Find saved template on screen and click it."""
    template_path = TEMPLATES_DIR / f"{name}.png"
    meta_path = TEMPLATES_DIR / f"{name}.json"
    
    if not template_path.exists():
        return {"success": False, "error": f"Template not found: {name}. Run: zoomclick --list"}
    
    # Load metadata for fallback coordinates
    meta = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
    
    # Take screenshot
    screenshot_path = take_screenshot("click")
    
    # Try to find template
    match = find_template_on_screen(template_path, screenshot_path)
    
    if match:
        x, y, conf = match
        method = "template_match"
    else:
        # Fallback to saved coordinates
        x = meta.get("center_x")
        y = meta.get("center_y")
        conf = 0.0
        method = "saved_coords"
        
        if x is None or y is None:
            return {
                "success": False,
                "error": f"Could not find template on screen and no saved coordinates",
                "screenshot": str(screenshot_path)
            }
    
    if not no_click:
        pyautogui.moveTo(x, y, duration=0.25)
        pyautogui.click(x, y)
    
    return {
        "success": True,
        "action": "click" if not no_click else "locate",
        "name": name,
        "x": x,
        "y": y,
        "confidence": round(conf, 3) if conf else None,
        "method": method,
        "screenshot": str(screenshot_path)
    }

def click_center(no_click: bool = False) -> dict:
    """Click the center of current viewport without saving."""
    state = ViewportState.load()
    if not state:
        return {"success": False, "error": "No active session. Run: zoomclick --start"}
    
    # Viewport-relative center
    viewport_x = state.x + state.width // 2
    viewport_y = state.y + state.height // 2
    
    # Screen-absolute coordinates (account for window offset)
    screen_x = viewport_x + state.window_offset_x
    screen_y = viewport_y + state.window_offset_y
    
    if not no_click:
        pyautogui.moveTo(screen_x, screen_y, duration=0.25)
        pyautogui.click(screen_x, screen_y)
    
    # Take screenshot after click
    if state.window_id:
        screenshot_path = take_screenshot_window(state.window_id, "after_click")
    else:
        screenshot_path = take_screenshot("after_click")
    
    result = {
        "success": True,
        "action": "click" if not no_click else "locate",
        "viewport_coords": {"x": viewport_x, "y": viewport_y},
        "screen_coords": {"x": screen_x, "y": screen_y},
        "viewport": state.to_dict(),
        "screenshot": str(screenshot_path)
    }
    
    if state.window_id:
        result["window_id"] = state.window_id
    
    return result

def list_templates() -> dict:
    """List all saved templates."""
    templates = []
    
    for png in sorted(TEMPLATES_DIR.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True):
        name = png.stem
        meta_path = TEMPLATES_DIR / f"{name}.json"
        
        meta = {}
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
        
        templates.append({
            "name": name,
            "base_name": meta.get("base_name", name),
            "path": str(png),
            "click_coords": {"x": meta.get("center_x"), "y": meta.get("center_y")},
            "created": meta.get("created")
        })
    
    return {
        "success": True,
        "action": "list",
        "templates": templates,
        "count": len(templates),
        "templates_dir": str(TEMPLATES_DIR)
    }

def reset_session() -> dict:
    """Reset zoom state."""
    if STATE_FILE.exists():
        STATE_FILE.unlink()
    
    return {
        "success": True,
        "action": "reset",
        "message": "Session reset. Run: zoomclick --start"
    }

def delete_template(name: str) -> dict:
    """Delete a saved template."""
    template_path = TEMPLATES_DIR / f"{name}.png"
    meta_path = TEMPLATES_DIR / f"{name}.json"
    
    deleted = []
    if template_path.exists():
        template_path.unlink()
        deleted.append(str(template_path))
    if meta_path.exists():
        meta_path.unlink()
        deleted.append(str(meta_path))
    
    if deleted:
        return {"success": True, "action": "delete", "name": name, "deleted": deleted}
    else:
        return {"success": False, "error": f"Template not found: {name}"}

def main():
    parser = argparse.ArgumentParser(
        description="Iterative zoom-and-click tool for AI-assisted UI automation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--start", "-s", action="store_true", help="Start new session with full screenshot")
    group.add_argument("--zoom", "-z", metavar="QUADRANT", help="Zoom into quadrant (top-left, top-right, bottom-left, bottom-right, center)")
    group.add_argument("--save", metavar="NAME", help="Save current view as named template")
    group.add_argument("--click", "-c", metavar="NAME", help="Find and click saved template")
    group.add_argument("--click-center", action="store_true", help="Click center of current viewport")
    group.add_argument("--list", "-l", action="store_true", help="List saved templates")
    group.add_argument("--reset", "-r", action="store_true", help="Reset zoom session")
    group.add_argument("--delete", "-d", metavar="NAME", help="Delete saved template")
    group.add_argument("--list-windows", action="store_true", help="List all visible windows")
    
    parser.add_argument("--no-click", action="store_true", help="Don't click, just locate")
    parser.add_argument("--display", default=":99", help="X display (default :99)")
    
    # Window/screen targeting options (used with --start)
    parser.add_argument("--window", "-w", help="Capture window by title (substring match)")
    parser.add_argument("--window-class", help="Capture window by class name")
    parser.add_argument("--window-id", type=int, help="Capture specific window ID")
    parser.add_argument("--screen", type=int, help="Capture specific screen number (0-indexed)")
    
    args = parser.parse_args()
    
    os.environ['DISPLAY'] = args.display
    
    try:
        # Handle list-windows first (doesn't need session)
        if args.list_windows:
            windows = list_windows()
            result = {
                "success": True,
                "action": "list_windows",
                "windows": windows,
                "count": len(windows)
            }
            print(json.dumps(result, indent=2))
            return 0
        
        if args.start:
            # Resolve window targeting
            window_id = None
            if args.window_id:
                window_id = args.window_id
            elif args.window:
                windows = find_window_by_name(args.window)
                if not windows:
                    print(json.dumps({"success": False, "error": f"No window found matching: {args.window}"}))
                    return 1
                window_id = windows[0]
            elif args.window_class:
                windows = find_window_by_class(args.window_class)
                if not windows:
                    print(json.dumps({"success": False, "error": f"No window found with class: {args.window_class}"}))
                    return 1
                window_id = windows[0]
            
            result = start_session(window_id=window_id, screen_num=args.screen)
        elif args.zoom:
            result = zoom_to_quadrant(args.zoom)
        elif args.save:
            result = save_template(args.save)
        elif args.click:
            result = click_template(args.click, args.no_click)
        elif args.click_center:
            result = click_center(args.no_click)
        elif args.list:
            result = list_templates()
        elif args.reset:
            result = reset_session()
        elif args.delete:
            result = delete_template(args.delete)
        else:
            parser.print_help()
            return 1
        
        print(json.dumps(result, indent=2))
        return 0 if result.get("success", True) else 1
        
    except Exception as e:
        print(json.dumps({"success": False, "error": str(e)}))
        return 1

if __name__ == "__main__":
    sys.exit(main())
