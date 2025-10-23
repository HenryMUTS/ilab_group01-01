import cv2
import numpy as np
import os

# ---------------------------------------------------
# Hardcoded directories (EDIT THESE)
# ---------------------------------------------------
pred_dir     = "/Users/atyantjain/Desktop/MDSI/Ilab Capstone/New/CodeFormer/inputs/whole_imgs"
upscaled_dir = "/Users/atyantjain/Desktop/MDSI/Ilab Capstone/New/CodeFormer/Final_result/final_results"
mask_dir     = "/Users/atyantjain/Desktop/MDSI/Ilab Capstone/New/mask_input_new"
out_dir      = "/Users/atyantjain/Desktop/MDSI/Ilab Capstone/New/outputs/final_merge"

os.makedirs(out_dir, exist_ok=True)

# ---------------------------------------------------
# Helper functions
# ---------------------------------------------------
def load_image(path, flag=cv2.IMREAD_COLOR):
    img = cv2.imread(path, flag)
    if img is None:
        raise FileNotFoundError(f"❌ Could not read image: {path}")
    return img

def preprocess_mask(mask_gray, target_shape):
    mask = cv2.resize(mask_gray, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
    mask = mask.astype(np.float32) / 255.0
    # Normalize and feather
    mask = cv2.normalize(mask, None, 0, 1, cv2.NORM_MINMAX)
    mask = cv2.GaussianBlur(mask, (21, 21), 10)
    mask = np.clip(mask, 0, 1)
    # Convert to 3-channel
    mask_3c = np.repeat(mask[..., None], 3, axis=2)
    return mask_3c

# ---------------------------------------------------
# Loop through prediction images
# ---------------------------------------------------
for fname in os.listdir(pred_dir):
    if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    pred_path     = os.path.join(pred_dir, fname)
    upscaled_path = os.path.join(upscaled_dir, fname)
    mask_path     = os.path.join(mask_dir, fname)
    out_path      = os.path.join(out_dir, fname)

    if not (os.path.exists(upscaled_path) and os.path.exists(mask_path)):
        print(f"⚠️ Missing match for {fname}, skipping.")
        continue

    # Load images
    pred = load_image(pred_path)
    upscaled = load_image(upscaled_path)
    mask_gray = load_image(mask_path, cv2.IMREAD_GRAYSCALE)

    # Resize to prediction size
    h, w = pred.shape[:2]
    upscaled = cv2.resize(upscaled, (w, h))
    mask_3c = preprocess_mask(mask_gray, pred.shape)

    # Detect if mask is inverted (white background, black nose)
    nose_pixels = np.mean(mask_3c[:, :, 0] > 0.5)
    if nose_pixels > 0.7:
        print(f"↻ Inverting mask for {fname}")
        mask_3c = 1.0 - mask_3c

    # ---------------------------------------------------
    # Correct blending: prediction face + upscaled nose
    # ---------------------------------------------------
    merged = pred.astype(np.float32) * (1 - mask_3c) + upscaled.astype(np.float32) * mask_3c
    merged = np.clip(merged, 0, 255).astype(np.uint8)

    # Save output
    cv2.imwrite(out_path, merged)
    print(f"✅ Saved merged image: {out_path}")

print("\n🎉 All images merged successfully! Face from prediction + nose from upscaled.")
