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

def _dilate_mask(mask_t: torch.Tensor, dilate_px: int) -> torch.Tensor:
    """Dilate a binary/soft mask [B,1,H,W] by `dilate_px` pixels."""
    if dilate_px <= 0:
        return mask_t
    k = dilate_px * 2 + 1
    return F.max_pool2d(mask_t, kernel_size=k, stride=1, padding=dilate_px)

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

def generate_mask_from_image(image: Image.Image, dilate_px=35) -> Image.Image:
    """Generate a face-aligned nose mask using face-alignment landmarks."""
    img_rgb = np.array(image.convert("RGB"))
    landmarks, found = mask_generator.detect_nose_landmarks(img_rgb)

    if not found:
        print("⚠️ No face/nose landmarks detected — returning blank mask.")
        mask = np.zeros((img_rgb.shape[0], img_rgb.shape[1]), dtype=np.float32)
    else:
        mask = mask_generator.create_mask(img_rgb, landmarks, dilate_px=dilate_px)

    # Convert to PIL grayscale image for further processing
    mask_img = Image.fromarray((mask * 255).astype(np.uint8))
    return mask_img


# ==========================================================
#  Inference
# ==========================================================

@torch.no_grad()
def infer_single(G, img_bytes, dilate_px=0, device=None):
    # if device is None:
    #     device = next(G.parameters()).device

    # # --- Load and preprocess ---
    # rgb = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    # rgb = rgb.resize((512, 512))  # ✅ match training resolution
    # mask = generate_mask_from_image(rgb)

    # transform = T.Compose([
    #     T.ToTensor(),
    #     T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # ✅ scale to [-1, 1]
    # ])
    # mask_t = T.ToTensor()(mask)
    # rgb_t = transform(rgb)
    # inp = torch.cat([rgb_t, mask_t], dim=0).unsqueeze(0).to(device)

    # # --- Run inference ---
    # out, full_rgb, pred_mask = G(inp, return_full=True)

    # # --- Denormalize output ---
    # out = (out * 0.5 + 0.5).clamp(0, 1)  # ✅ scale back to [0,1]

    if device is None:
        device = next(G.parameters()).device

    # ---- Load RGB ----
    rgb = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    rgb_t = T.ToTensor()(rgb).unsqueeze(0).to(device)  # [1,3,H,W]

    # ---- Load mask ----
    mask = generate_mask_from_image(rgb)
    mask_t = T.ToTensor()(mask).unsqueeze(0).to(device)
    if mask_t.shape[-2:] != rgb_t.shape[-2:]:
        mask_t = F.interpolate(mask_t, size=rgb_t.shape[-2:], mode="nearest")

    # ---- Forward ----
    inp = torch.cat([rgb_t, mask_t], dim=1)
    try:
        out, full_rgb, pred_mask = G(inp, return_full=True)
    except TypeError:
        full_rgb = G(inp)
        pred_mask = mask_t

    # ---- Dilate & blend ----
    mask_d = _dilate_mask(pred_mask, dilate_px).clamp(0, 1)
    blended = rgb_t + mask_d * (full_rgb - rgb_t)
    out = blended.clamp(0, 1)


    # --- Convert to image ---
    out_buf = io.BytesIO()
    save_image(out, out_buf, format="PNG")
    out_buf.seek(0)
    return out_buf

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
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    out_buf = infer_single(G, img_bytes, dilate_px=0, device=device)
    out_buf.seek(0)
    return StreamingResponse(out_buf, media_type="image/png")



@app.get("/")
def home():
    return {"message": "UNet PatchGAN Nose Prediction API is running (single image input)."}
