"""
Utility to load, process, and provide base64 data URIs for the user's SkyGuard AI logo.
Supports generating dark-mode optimized transparent PNG so it blends seamlessly
with the charcoal black enterprise theme.
"""

import base64
import io
import os
from PIL import Image
import numpy as np

USER_LOGO_SRC = "/Users/suryansh/.gemini/antigravity-ide/brain/ab263a68-6185-4daa-a4bc-3894b9af91c9/.user_uploaded/media_1790098225133.png"
ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
DARK_LOGO_PATH = os.path.join(ASSETS_DIR, "skyguard_logo_dark.png")
ORIG_LOGO_PATH = os.path.join(ASSETS_DIR, "skyguard_logo_transparent.png")

def ensure_logo_assets():
    """Generates transparent, dark-mode-ready and original-color logos."""
    os.makedirs(ASSETS_DIR, exist_ok=True)
    if os.path.exists(DARK_LOGO_PATH) and os.path.exists(ORIG_LOGO_PATH):
        return

    if not os.path.exists(USER_LOGO_SRC):
        return

    try:
        img = Image.open(USER_LOGO_SRC).convert("RGBA")
        arr = np.array(img, dtype=np.float32)

        # Background color in the uploaded image is off-white [249, 249, 247]
        bg_color = np.array([249.0, 249.0, 247.0])

        # Logo text bounding box: y=455..568, x=105..920
        cropped = arr[455:568, 105:920].copy()
        h, w, _ = cropped.shape

        # Distance from background color
        dist = np.sqrt(np.sum((cropped[:, :, :3] - bg_color) ** 2, axis=2))

        # Alpha mask based on distance (smooth anti-aliasing)
        alpha = np.clip((dist - 8.0) / (72.0 - 8.0), 0.0, 1.0) * 255.0

        # Detect cyan 'AI' region where blue channel is higher than red
        is_cyan_mask = (cropped[:, :, 2] - cropped[:, :, 0]) > 25.0

        # Build dark-mode version: white/silver text for SKYGUARD + vibrant cyan for AI
        dark_ui = np.zeros_like(cropped)
        for y in range(h):
            for x in range(w):
                a = alpha[y, x]
                if a < 4.0:
                    dark_ui[y, x] = [0, 0, 0, 0]
                elif is_cyan_mask[y, x]:
                    # Cyan 'AI' part: keep vibrant cyan #38BDF8 / electric teal
                    orig_r, orig_g, orig_b = cropped[y, x, :3]
                    # Map to brilliant cyan
                    dark_ui[y, x] = [40, 185, 235, a]
                else:
                    # 'SKYGUARD' letters & shield: clean crisp white #F8FAFC
                    dark_ui[y, x] = [248, 250, 252, a]

        dark_img = Image.fromarray(dark_ui.astype(np.uint8))
        dark_img.save(DARK_LOGO_PATH, format="PNG")

        # Also create transparent original colors
        orig_ui = cropped.copy()
        orig_ui[:, :, 3] = alpha
        orig_img = Image.fromarray(orig_ui.astype(np.uint8))
        orig_img.save(ORIG_LOGO_PATH, format="PNG")
    except Exception as e:
        print(f"Error processing logo: {e}")


def get_logo_base64(mode="dark"):
    """Returns a base64 data URI string for embedding directly into HTML/CSS."""
    ensure_logo_assets()
    target_path = DARK_LOGO_PATH if mode == "dark" else ORIG_LOGO_PATH
    if not os.path.exists(target_path):
        if os.path.exists(USER_LOGO_SRC):
            target_path = USER_LOGO_SRC
        else:
            return ""

    with open(target_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:image/png;base64,{b64}"
