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

Usage Examples:
  zoomclick --start                    # Take screenshot, show with quadrant guides
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

# Directories
WORK_DIR = Path("/tmp/zoomclick")
TEMPLATES_DIR = Path.home() / ".zoomclick" / "templates"
STATE_FILE = WORK_DIR / "state.json"

WORK_DIR.mkdir(exist_ok=True)
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)

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

def take_screenshot(name="screen") -> Path:
    """Take a full screenshot using scrot."""
    timestamp = int(time.time())
    path = WORK_DIR / f"{name}_{timestamp}.png"
    
    display = os.environ.get('DISPLAY', ':99')
    result = subprocess.run(
        ['scrot', str(path)],
        env={**os.environ, 'DISPLAY': display},
        capture_output=True
    )
    
    if result.returncode != 0:
        raise RuntimeError(f"Screenshot failed: {result.stderr.decode()}")
    
    return path

def get_screen_size() -> Tuple[int, int]:
    """Get screen dimensions."""
    return pyautogui.size()

def crop_image(src_path: Path, x: int, y: int, width: int, height: int, dst_path: Path):
    """Crop image using ImageMagick convert."""
    subprocess.run([
        'convert', str(src_path),
        '-crop', f'{width}x{height}+{x}+{y}',
        '+repage',
        str(dst_path)
    ], check=True)

def add_quadrant_overlay(src_path: Path, dst_path: Path, width: int, height: int):
    """Add quadrant guide lines to image for AI visualization."""
    # Draw grid lines: 2 vertical, 2 horizontal (dividing into 9 regions - 4 corners + 4 edges + center)
    # For simplicity, we'll divide into 4 quadrants + center
    mid_x = width // 2
    mid_y = height // 2
    quarter_x = width // 4
    quarter_y = height // 4
    
    # Create semi-transparent overlay with quadrant labels
    subprocess.run([
        'convert', str(src_path),
        # Vertical center line
        '-stroke', 'rgba(255,0,0,0.5)', '-strokewidth', '2',
        '-draw', f'line {mid_x},0 {mid_x},{height}',
        # Horizontal center line  
        '-draw', f'line 0,{mid_y} {width},{mid_y}',
        # Center region box (inner 50%)
        '-stroke', 'rgba(0,255,0,0.5)', '-strokewidth', '2',
        '-fill', 'none',
        '-draw', f'rectangle {quarter_x},{quarter_y} {width-quarter_x},{height-quarter_y}',
        # Labels
        '-fill', 'rgba(255,255,255,0.9)', '-stroke', 'none',
        '-font', 'DejaVu-Sans', '-pointsize', '20',
        '-gravity', 'NorthWest', '-annotate', '+10+10', 'TOP-LEFT',
        '-gravity', 'NorthEast', '-annotate', '+10+10', 'TOP-RIGHT',
        '-gravity', 'SouthWest', '-annotate', '+10+10', 'BOTTOM-LEFT',
        '-gravity', 'SouthEast', '-annotate', '+10+10', 'BOTTOM-RIGHT',
        '-gravity', 'Center', '-annotate', '+0+0', 'CENTER',
        str(dst_path)
    ], check=True, capture_output=True)

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

def start_session() -> dict:
    """Start a new zoom session with full screenshot."""
    screenshot_path = take_screenshot("full")
    screen_w, screen_h = get_screen_size()
    
    # Initialize viewport state
    state = ViewportState(
        x=0, y=0,
        width=screen_w, height=screen_h,
        screen_width=screen_w, screen_height=screen_h,
        zoom_level=0
    )
    state.save()
    
    # Create overlay version
    overlay_path = WORK_DIR / f"overlay_{int(time.time())}.png"
    add_quadrant_overlay(screenshot_path, overlay_path, screen_w, screen_h)
    
    return {
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

def zoom_to_quadrant(quadrant: str) -> dict:
    """Zoom into a quadrant of the current viewport."""
    state = ViewportState.load()
    if not state:
        return {"success": False, "error": "No active session. Run: zoomclick --start"}
    
    # Get new viewport bounds
    new_x, new_y, new_w, new_h = get_quadrant_bounds(state, quadrant)
    
    # Take fresh screenshot
    screenshot_path = take_screenshot("full")
    
    # Crop to new viewport
    cropped_path = WORK_DIR / f"zoom_{state.zoom_level + 1}_{int(time.time())}.png"
    crop_image(screenshot_path, new_x, new_y, new_w, new_h, cropped_path)
    
    # Update state
    state.x = new_x
    state.y = new_y
    state.width = new_w
    state.height = new_h
    state.zoom_level += 1
    state.save()
    
    # Add overlay to cropped image
    overlay_path = WORK_DIR / f"overlay_{state.zoom_level}_{int(time.time())}.png"
    add_quadrant_overlay(cropped_path, overlay_path, new_w, new_h)
    
    return {
        "success": True,
        "action": "zoom",
        "quadrant": quadrant,
        "screenshot": str(overlay_path),
        "viewport": state.to_dict(),
        "screen_coords": {
            "center_x": new_x + new_w // 2,
            "center_y": new_y + new_h // 2,
            "description": "If you clicked now, this would be the screen coordinate"
        },
        "instructions": f"""
Zoomed into {quadrant}. Now at zoom level {state.zoom_level}.
Viewport: {new_w}x{new_h} at screen position ({new_x}, {new_y})

If your target is BIG and CENTERED, save it:
  zoomclick --save "button_name"

If you need to zoom more, pick another quadrant:
  zoomclick --zoom <quadrant>

To click the current center immediately (without saving):
  zoomclick --click-center
""".strip()
    }

def save_template(name: str) -> dict:
    """Save current viewport as a reusable template."""
    state = ViewportState.load()
    if not state:
        return {"success": False, "error": "No active session. Run: zoomclick --start"}
    
    # Take fresh screenshot and crop current viewport
    screenshot_path = take_screenshot("full")
    
    template_path = TEMPLATES_DIR / f"{name}.png"
    meta_path = TEMPLATES_DIR / f"{name}.json"
    
    crop_image(screenshot_path, state.x, state.y, state.width, state.height, template_path)
    
    # Save metadata - the center coordinates to click
    center_x = state.x + state.width // 2
    center_y = state.y + state.height // 2
    
    meta = {
        "name": name,
        "center_x": center_x,
        "center_y": center_y,
        "viewport": state.to_dict(),
        "created": time.time(),
        "note": "Template saved for future clicking. Use: zoomclick --click " + name
    }
    
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    
    return {
        "success": True,
        "action": "save",
        "name": name,
        "template_path": str(template_path),
        "click_coords": {"x": center_x, "y": center_y},
        "instructions": f"""
Template "{name}" saved!
- Image: {template_path}
- Click coordinates: ({center_x}, {center_y})

To click this element anytime, run:
  zoomclick --click "{name}"

The tool will find the template on screen and click its center.
""".strip()
    }

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
    
    x = state.x + state.width // 2
    y = state.y + state.height // 2
    
    if not no_click:
        pyautogui.moveTo(x, y, duration=0.25)
        pyautogui.click(x, y)
    
    # Take screenshot after click
    screenshot_path = take_screenshot("after_click")
    
    return {
        "success": True,
        "action": "click" if not no_click else "locate",
        "x": x,
        "y": y,
        "viewport": state.to_dict(),
        "screenshot": str(screenshot_path)
    }

def list_templates() -> dict:
    """List all saved templates."""
    templates = []
    
    for png in TEMPLATES_DIR.glob("*.png"):
        name = png.stem
        meta_path = TEMPLATES_DIR / f"{name}.json"
        
        meta = {}
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
        
        templates.append({
            "name": name,
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
    
    parser.add_argument("--no-click", action="store_true", help="Don't click, just locate")
    parser.add_argument("--display", default=":99", help="X display (default :99)")
    
    args = parser.parse_args()
    
    os.environ['DISPLAY'] = args.display
    
    try:
        if args.start:
            result = start_session()
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
