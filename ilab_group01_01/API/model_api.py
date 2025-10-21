from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
from torchvision import transforms as T
from PIL import Image, ImageDraw
import io, base64
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from lpips import LPIPS
from typing import Optional
from fastapi.responses import StreamingResponse
import cv2
import numpy as np
import face_alignment
import zipfile
import subprocess

# ==========================================================
#  Functions Helpers
# ==========================================================

def _dilate_mask(mask_t: torch.Tensor, dilate_px: int) -> torch.Tensor:
    """Dilate a binary/soft mask [B,1,H,W] by `dilate_px` pixels."""
    if dilate_px <= 0:
        return mask_t
    k = dilate_px * 2 + 1
    return F.max_pool2d(mask_t, kernel_size=k, stride=1, padding=dilate_px)

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

# ==========================================================
#  UNet + Inference Utilities
# ==========================================================

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.SiLU(inplace=True),
        )
    def forward(self, x): return self.block(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_ch, out_ch)
    def forward(self, x): return self.conv(self.pool(x))

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = ConvBlock(in_ch, out_ch)
    def forward(self, x, skip):
        x = self.up(x)
        diffY = skip.size(2) - x.size(2)
        diffX = skip.size(3) - x.size(3)
        if diffY != 0 or diffX != 0:
            x = F.pad(x, (0, diffX, 0, diffY))
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)

class UNetNoseGenerator(nn.Module):
    """
    UNet that takes RGB+mask (4ch) and predicts a 3ch residual 'delta'.
    Output = rgb + soft_mask * tanh(raw) * scale
    """
    def __init__(self, in_ch=4, out_ch=3, base=64, depth=5, res_max=1):
        """
        depth=5 -> downsample x2 five times (stride 32). Use collate multiple=32.
        For less memory, set depth=4 (stride 16).
        """
        super().__init__()
        self.out_ch = out_ch
        self.res_max = res_max

        # encoder
        self.inc  = ConvBlock(in_ch, base)                 # H
        self.down1 = Down(base,     base*2)                # H/2
        self.down2 = Down(base*2,   base*4)                # H/4
        self.down3 = Down(base*4,   base*8)                # H/8
        self.down4 = Down(base*8,   base*8)                # H/16
        self.has_down5 = (depth >= 5)
        if self.has_down5:
            self.down5 = Down(base*8, base*8)              # H/32

        # decoder
        if self.has_down5:
            self.up1 = Up(base*8 + base*8, base*8)         # concat with down4
            ch_up_in = base*8 + base*8
        else:
            # if no down5, first up will concatenate down3 and bottleneck at base*8
            ch_up_in = base*8 + base*8  # consistent with next lines

        self.up2 = Up(base*8 + base*8, base*8)             # + down3
        self.up3 = Up(base*8 + base*4, base*4)             # + down2
        self.up4 = Up(base*4 + base*2, base*2)             # + down1
        self.up5 = Up(base*2 + base,   base)               # + inc

        self.outc = nn.Conv2d(base, out_ch, kernel_size=3, padding=1)
        nn.init.zeros_(self.outc.weight)
        nn.init.zeros_(self.outc.bias)

        # learnable residual cap
        self._alpha = nn.Parameter(torch.tensor(0.0))  # sigmoid ~0.5 initially

    def forward(self, inp, return_full=False):
        """
        inp: [B,4,H,W]  (RGB + binary/soft mask in channel 4)
        returns: [B,3,H,W] blended output at original size
        """
        rgb   = inp[:, :3]
        mask1 = inp[:, 3:4]

        # encoder
        x1 = self.inc(inp)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        if self.has_down5:
            x6 = self.down5(x5)
            u1 = self.up1(x6, x5)
        else:
            u1 = self.up2(x5, x4)  # skip one level if depth=4

        # decoder path
        if self.has_down5:
            u2 = self.up2(u1, x4)
        else:
            u2 = u1
        u3 = self.up3(u2, x3)
        u4 = self.up4(u3, x2)
        u5 = self.up5(u4, x1)

        raw   = self.outc(u5)               # [B,3,H,W]

        delta = raw
        H, W = rgb.shape[-2:]

        if delta.shape[-2:] != (H, W):

            delta = F.interpolate(delta, size=(H, W), mode='bilinear', align_corners=False)
        hard   = mask1
        if hard.shape[-2:] != (H, W):
            hard = F.interpolate(hard, size=(H, W), mode='nearest').clamp(0, 1)

        # soft blend within a feathered mask band

        #m_soft = feather_mask(hard, k=9)            # [B,1,H,W]
        m_soft = hard
        m3     = m_soft.repeat(1, 3, 1, 1)
        full_rgb = (rgb + delta)
        out    = rgb + delta * m3
        if return_full:
            return out, full_rgb, m_soft
        return out


# ==========================================================
#  Mask Generator (placeholder)
# ==========================================================

class FANNoseMaskGenerator:
    def __init__(self, device=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_D,
            flip_input=False,
            device=device
        )
        self.nose_idx = list(range(27, 36))  # 27–35 inclusive

    def detect_nose_landmarks(self, img_rgb):
        lms = self.fa.get_landmarks(img_rgb)
        if lms is None or len(lms) == 0:
            return [], False
        nose_pts = lms[0][self.nose_idx, :2].astype(int).tolist()
        return nose_pts, True

    def create_mask(self, img_rgb, landmarks, dilate_px=35, up_shift=8, right_shift=6):
        """
        1. Build convex-hull mask from nose landmarks.
        2. Dilate isotropically (dilate_px).
        3. Expand slightly upward and right (directional bias).
        """
        H, W = img_rgb.shape[:2]
        mask = np.zeros((H, W), dtype=np.float32)
        if not landmarks:
            return mask

        # ---- base hull mask ----
        pts = np.array(landmarks, dtype=np.int32)
        hull = cv2.convexHull(pts)
        cv2.fillConvexPoly(mask, hull, 1.0)

        # ---- isotropic dilation ----
        if dilate_px > 0:
            k = dilate_px if dilate_px % 2 == 1 else dilate_px + 1
            ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            mask = cv2.dilate(mask, ker, iterations=1)

        # ---- directional expansion ----
        m = mask.copy()
        if up_shift > 0:
            roll_up = np.roll(mask, -up_shift, axis=0)
            roll_up[-up_shift:, :] = 0
            m = np.maximum(m, roll_up)
        if right_shift > 0:
            roll_right = np.roll(mask, right_shift, axis=1)
            roll_right[:, :right_shift] = 0
            m = np.maximum(m, roll_right)

        return np.clip(m, 0, 1).astype(np.float32)

    def visualize(self, img_rgb, mask, landmarks=None):
        overlay = img_rgb.copy()
        if landmarks:
            for p in landmarks:
                cv2.circle(overlay, p, 2, (0,255,0), -1)
        blend = (0.7 * overlay + 0.3 * (mask[...,None]*np.array([255,0,0]))).astype(np.uint8)
        return blend

device = 'cuda' if torch.cuda.is_available() else 'cpu'
mask_generator = FANNoseMaskGenerator(device=device)

def generate_mask_from_image(image: Image.Image, dilate_px=35) -> tuple[Image.Image, FANNoseMaskGenerator]:
    """Generate a face-aligned nose mask using face-alignment landmarks."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mask_generator = FANNoseMaskGenerator(device=device)

    img_rgb = np.array(image.convert("RGB"))
    landmarks, found = mask_generator.detect_nose_landmarks(img_rgb)

    if not found:
        print("⚠️ No face/nose landmarks detected — returning blank mask.")
        mask = np.zeros((img_rgb.shape[0], img_rgb.shape[1]), dtype=np.float32)
    else:
        mask = mask_generator.create_mask(img_rgb, landmarks, dilate_px=dilate_px)

    mask_img = Image.fromarray((mask * 255).astype(np.uint8))
    return mask_img, mask_generator


# ==========================================================
#  Inference
# ==========================================================

@torch.no_grad()
def infer_single(G, img_bytes, mask_bytes=None, dilate_px=0, device=None):
    if device is None:
        device = next(G.parameters()).device

    # ---- Load RGB image ----
    rgb = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    rgb_t = T.ToTensor()(rgb).unsqueeze(0).to(device)

    # ---- Load or generate mask ----
    if mask_bytes:  # Custom user-provided mask
        print("🟩 Using user-provided custom mask.")
        mask = Image.open(io.BytesIO(mask_bytes)).convert("L")
        mask_generator = None  # No need to visualize landmarks
    else:  # Auto-generate mask via face landmarks
        print("🟦 Auto-generating face-aligned mask.")
        mask, mask_generator = generate_mask_from_image(rgb)

    # ---- Convert mask to tensor ----
    mask_t = T.ToTensor()(mask).unsqueeze(0).to(device)
    if mask_t.shape[-2:] != rgb_t.shape[-2:]:
        mask_t = F.interpolate(mask_t, size=rgb_t.shape[-2:], mode="nearest")

    # ---- Forward pass through UNet ----
    inp = torch.cat([rgb_t, mask_t], dim=1)
    try:
        out, full_rgb, pred_mask = G(inp, return_full=True)
    except TypeError:
        full_rgb = G(inp)
        pred_mask = mask_t

    # ---- Blend with mask ----
    mask_d = _dilate_mask(pred_mask, dilate_px).clamp(0, 1)
    blended = rgb_t + mask_d * (full_rgb - rgb_t)
    out = blended.clamp(0, 1)

    # --- Convert prediction to image ---
    pred_buf = io.BytesIO()
    save_image(out, pred_buf, format="PNG")
    pred_buf.seek(0)

    # --- Create mask overlay visualization ---
    img_rgb = np.array(rgb.convert("RGB"))
    mask_np = np.array(mask.resize(rgb.size))

    if mask_generator is not None:
        overlay = mask_generator.visualize(img_rgb, mask_np)
    else:
        overlay = (0.7 * img_rgb + 0.3 * (mask_np[..., None] * np.array([255, 0, 0]))).astype(np.uint8)

    overlay_buf = io.BytesIO()
    Image.fromarray(overlay).save(overlay_buf, format="PNG")
    overlay_buf.seek(0)

    return pred_buf, overlay_buf

# ==========================================================
#  FastAPI Setup
# ==========================================================

app = FastAPI(title="UNet Prediction API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# ------------ Load pretrained model ---------------
device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "C:/Users/henry/OneDrive/Desktop/UTS/iLab/Model/Practical/best_l1_mask_latest.pt"

G = UNetNoseGenerator(in_ch=4, out_ch=3, base=64, depth=5, res_max=0.75).to(device)
ckpt = torch.load(MODEL_PATH, map_location=device)
if "G" in ckpt:
    G.load_state_dict(ckpt["G"], strict=False)
else:
    G.load_state_dict(ckpt, strict=False)
G.eval()

print(f"✅ Model loaded from {MODEL_PATH} on {device}")

# ==========================================================
#  Endpoints
# ==========================================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    mask_file: Optional[UploadFile] = File(None)
):
    img_bytes = await file.read()
    mask_bytes = await mask_file.read() if mask_file else None

    # Run inference → get prediction & mask overlay
    pred_buf, overlay_buf = infer_single(G, img_bytes, mask_bytes, dilate_px=0, device=device)

    # SAVE TO LOCAL DIRECTORY
    pred_save_dir = "models/CodeFormer/inputs/whole_imgs"  # change as needed
    mask_save_dir = "data/interim"

    import os
    os.makedirs(pred_save_dir, exist_ok=True)  # create folder if not exists
    os.makedirs(mask_save_dir, exist_ok=True)

    # Use same filename (without extension) for consistency
    base_name = os.path.splitext(file.filename)[0]

    pred_path = os.path.join(pred_save_dir, f"{base_name}_prediction.png")
    mask_path = os.path.join(mask_save_dir, f"{base_name}_mask.png")
    
    # Write from buffers to disk
    with open(pred_path, "wb") as f:
        f.write(pred_buf.getvalue())
    with open(mask_path, "wb") as f:
        f.write(overlay_buf.getvalue())

    print(f"✅ Saved prediction to: {pred_path}")
    print(f"✅ Saved mask overlay to: {mask_path}")

    codeformer_input_dir = os.path.abspath("models/CodeFormer/inputs/whole_imgs")
    codeformer_output_dir = os.path.abspath("models/CodeFormer/Final_result")

    # RUN EXTERNAL SCRIPT (CodeFormer inference)
    command = [
        "python", "models/CodeFormer/inference_codeformer.py",
        "-i", codeformer_input_dir,
        "-o", codeformer_output_dir,
        "-w", "1",
        "--face_upsample",
        "--bg_upsampler", "realesrgan",
        "--bg_tile", "400",
        "-s", "2"
    ]

    try:
        result = subprocess.run(
            command,
            check=True,          # raises CalledProcessError if script fails
            capture_output=True, # capture stdout and stderr
            text=True
        )
        print("✅ CodeFormer completed successfully:")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("❌ CodeFormer failed:")
        print(e.stderr)

    # Extract Post Process Image

    pred_path_name = os.path.basename(pred_path)
    upscaled_path = os.path.join("models/CodeFormer/Final_result/final_results", pred_path_name)

    pred = load_image(pred_path)
    upscaled = load_image(upscaled_path)
    mask_gray = load_image(mask_path, cv2.IMREAD_GRAYSCALE)

    h, w = pred.shape[:2]
    upscaled = cv2.resize(upscaled, (w, h))
    mask_3c = preprocess_mask(mask_gray, pred.shape)

    nose_pixels = np.mean(mask_3c[:, :, 0] > 0.5)
    if nose_pixels > 0.7:
        print(f"↻ Inverting mask for {pred_path_name}")
        mask_3c = 1.0 - mask_3c

    merged = pred.astype(np.float32) * (1 - mask_3c) + upscaled.astype(np.float32) * mask_3c
    merged = np.clip(merged, 0, 255).astype(np.uint8)

    merged_rgb = cv2.cvtColor(merged, cv2.COLOR_BGR2RGB)
    merged_buf = io.BytesIO()
    Image.fromarray(merged_rgb).save(merged_buf, format="PNG")
    merged_buf.seek(0)

    # --- Create zip in-memory ---
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as zf:
        zf.writestr("prediction.png", merged_buf.getvalue())
        zf.writestr("mask_overlay.png", overlay_buf.getvalue())
    zip_buf.seek(0)

    return StreamingResponse(
        zip_buf,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=results.zip"}
    )

@app.get("/")
def home():
    return {"message": "UNet PatchGAN Nose Prediction API is running (single image input)."}
