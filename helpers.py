"""
Helper functions for zoomclick - window/screen utilities and screenshot tools.

These are display-only utilities. Overlays are never captured or saved to templates.
"""

import os
import subprocess
import time
from pathlib import Path

# Directories
WORK_DIR = Path("/tmp/zoomclick")
TEMPLATES_DIR = Path.home() / ".zoomclick" / "templates"
STATE_FILE = WORK_DIR / "state.json"

WORK_DIR.mkdir(exist_ok=True)
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)


def find_window_by_name(name: str) -> list:
    """Find window IDs by title (substring match)."""
    try:
        result = subprocess.run(
            ['xdotool', 'search', '--name', name],
            capture_output=True, text=True
        )
        if result.returncode == 0 and result.stdout.strip():
            return [int(wid) for wid in result.stdout.strip().split('\n') if wid]
    except Exception:
        pass
    return []


def find_window_by_class(class_name: str) -> list:
    """Find window IDs by class."""
    try:
        result = subprocess.run(
            ['xdotool', 'search', '--class', class_name],
            capture_output=True, text=True
        )
        if result.returncode == 0 and result.stdout.strip():
            return [int(wid) for wid in result.stdout.strip().split('\n') if wid]
    except Exception:
        pass
    return []


def get_window_geometry(window_id: int) -> dict:
    """Get window position and size."""
    try:
        result = subprocess.run(
            ['xdotool', 'getwindowgeometry', '--shell', str(window_id)],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            geo = {}
            for line in result.stdout.strip().split('\n'):
                if '=' in line:
                    key, val = line.split('=', 1)
                    geo[key] = int(val) if val.isdigit() else val
            return geo
    except Exception:
        pass
    return {}


def list_windows() -> list:
    """List all windows with IDs, names, and geometry."""
    windows = []
    try:
        result = subprocess.run(
            ['xdotool', 'search', '--onlyvisible', '--name', ''],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            for wid in result.stdout.strip().split('\n'):
                if not wid:
                    continue
                wid = int(wid)
                name_result = subprocess.run(
                    ['xdotool', 'getwindowname', str(wid)],
                    capture_output=True, text=True
                )
                name = name_result.stdout.strip() if name_result.returncode == 0 else ""
                geo = get_window_geometry(wid)
                windows.append({
                    "id": wid,
                    "name": name,
                    "x": geo.get("X", 0),
                    "y": geo.get("Y", 0),
                    "width": geo.get("WIDTH", 0),
                    "height": geo.get("HEIGHT", 0)
                })
    except Exception:
        pass
    return windows


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


def take_screenshot_window(window_id: int, name="window") -> Path:
    """Take screenshot of a specific window using ImageMagick import."""
    timestamp = int(time.time())
    path = WORK_DIR / f"{name}_{timestamp}.png"
    
    display = os.environ.get('DISPLAY', ':99')
    result = subprocess.run(
        ['import', '-window', str(window_id), str(path)],
        env={**os.environ, 'DISPLAY': display},
        capture_output=True
    )
    
    if result.returncode != 0:
        raise RuntimeError(f"Window screenshot failed: {result.stderr.decode()}")
    
    return path


def take_screenshot_screen(screen_num: int, name="screen") -> Path:
    """Take screenshot of a specific screen (multi-monitor)."""
    import re
    timestamp = int(time.time())
    path = WORK_DIR / f"{name}_{timestamp}.png"
    
    display = os.environ.get('DISPLAY', ':99')
    
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
    
    if screen_num >= len(screens):
        raise RuntimeError(f"Screen {screen_num} not found. Available: 0-{len(screens)-1}")
    
    s = screens[screen_num]
    full_path = WORK_DIR / f"full_{timestamp}.png"
    subprocess.run(['scrot', str(full_path)], env={**os.environ, 'DISPLAY': display}, check=True)
    
    subprocess.run([
        'convert', str(full_path),
        '-crop', f"{s['width']}x{s['height']}+{s['x']}+{s['y']}",
        '+repage', str(path)
    ], check=True)
    
    full_path.unlink()
    return path


def crop_image(src_path: Path, x: int, y: int, width: int, height: int, dst_path: Path):
    """Crop image using ImageMagick convert."""
    subprocess.run([
        'convert', str(src_path),
        '-crop', f'{width}x{height}+{x}+{y}',
        '+repage',
        str(dst_path)
    ], check=True)


def add_quadrant_overlay(src_path: Path, dst_path: Path, width: int, height: int):
    """
    Add quadrant guide lines to image for AI visualization.
    
    Note: Overlays are ONLY added to the output image for navigation.
    They never appear on the actual screen or in saved templates.
    """
    mid_x = width // 2
    mid_y = height // 2
    quarter_x = width // 4
    quarter_y = height // 4
    
    # Just draw guide lines, no text labels
    subprocess.run([
        'convert', str(src_path),
        # Vertical center line (red)
        '-stroke', 'rgba(255,0,0,0.5)', '-strokewidth', '2',
        '-draw', f'line {mid_x},0 {mid_x},{height}',
        # Horizontal center line (red)
        '-draw', f'line 0,{mid_y} {width},{mid_y}',
        # Center region box (green, inner 50%)
        '-stroke', 'rgba(0,255,0,0.5)', '-strokewidth', '2',
        '-fill', 'none',
        '-draw', f'rectangle {quarter_x},{quarter_y} {width-quarter_x},{height-quarter_y}',
        str(dst_path)
    ], check=True, capture_output=True)


def get_screen_size():
    """Get screen dimensions using pyautogui."""
    import pyautogui
    return pyautogui.size()
