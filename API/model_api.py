# api_unet_nose.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
import numpy as np
import cv2
from PIL import Image
import io
import face_alignment

# ---------------------------
# Model: UNetNoseGenerator
# ---------------------------
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
        self.up   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = ConvBlock(in_ch, out_ch)
    def forward(self, x, skip):
        x = self.up(x)
        diffY = skip.size(2) - x.size(2)
        diffX = skip.size(3) - x.size(3)
        if diffY != 0 or diffX != 0:
            x = F.pad(x, (0, diffX, 0, diffY))
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)

def dilate_mask_binary(mask, k=11, iters=2):
    out = mask
    for _ in range(iters):
        out = F.max_pool2d(out, kernel_size=k, stride=1, padding=k//2)
    return out.clamp(0,1)

def feather_mask(mask, k=9):
    return F.avg_pool2d(mask, kernel_size=k, stride=1, padding=k//2).clamp(0,1)

class UNetNoseGenerator(nn.Module):
    def __init__(self, in_ch=4, out_ch=3, base=64, depth=5, res_max=0.75):
        super().__init__()
        self.out_ch = out_ch
        self.res_max = res_max

        self.inc  = ConvBlock(in_ch, base)
        self.down1 = Down(base, base*2)
        self.down2 = Down(base*2, base*4)
        self.down3 = Down(base*4, base*8)
        self.down4 = Down(base*8, base*8)
        self.has_down5 = (depth >= 5)
        if self.has_down5:
            self.down5 = Down(base*8, base*8)

        if self.has_down5:
            self.up1 = Up(base*8 + base*8, base*8)
        self.up2 = Up(base*8 + base*8, base*8)
        self.up3 = Up(base*8 + base*4, base*4)
        self.up4 = Up(base*4 + base*2, base*2)
        self.up5 = Up(base*2 + base,   base)

        self.outc = nn.Conv2d(base, out_ch, 3, padding=1)
        nn.init.zeros_(self.outc.weight)
        nn.init.zeros_(self.outc.bias)

        self._alpha = nn.Parameter(torch.tensor(0.0))  

    def forward(self, inp, return_full=False):
        rgb   = inp[:, :3]
        mask1 = inp[:, 3:4]

        x1 = self.inc(inp)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        if self.has_down5:
            x6 = self.down5(x5)
            u1 = self.up1(x6, x5)
        else:
            u1 = self.up2(x5, x4)
        if self.has_down5:
            u2 = self.up2(u1, x4)
        else:
            u2 = u1
        u3 = self.up3(u2, x3)
        u4 = self.up4(u3, x2)
        u5 = self.up5(u4, x1)

        raw   = self.outc(u5)
        scale = torch.sigmoid(self._alpha) * self.res_max
        delta = torch.tanh(raw) * scale
        hard   = dilate_mask_binary(mask1, k=11, iters=2)
        m_soft = feather_mask(hard, k=9)
        m3     = m_soft.repeat(1, 3, 1, 1)
        full_rgb = (rgb + delta)
        out    = rgb + delta * m3
        if return_full:
            return out, full_rgb, m_soft
        return out

# ---------------------------
# Nose mask creation (face_alignment)
# ---------------------------
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device="cpu")

def create_nose_mask(image_np):
    preds = fa.get_landmarks(image_np)
    if preds is None: 
        return None
    landmarks = preds[0]
    nose_points = landmarks[27:36]
    mask = np.zeros(image_np.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.int32(nose_points), 255)
    mask = torch.from_numpy(mask).unsqueeze(0).float() / 255.0  # [1,H,W]
    return mask

# ---------------------------
# API setup
# ---------------------------
app = FastAPI()

model_art = "C:/Users/henry/OneDrive/Desktop/UTS/iLab/Model/Practical/Niketh_Unet_PatchGan_Model.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"
G = UNetNoseGenerator().to(device)
ckpt = torch.load(model_art, map_location=device)
G.load_state_dict(ckpt["G"], strict=True)
G.eval()

def preprocess_image(file: UploadFile):
    image = Image.open(io.BytesIO(file.file.read())).convert("RGB")
    img_np = np.array(image)
    rgb = torch.from_numpy(img_np).permute(2,0,1).float()/255.0
    return rgb, img_np

@app.post("/predict")
async def transform(file: UploadFile = File(...)):
    rgb, img_np = preprocess_image(file)
    mask = create_nose_mask(img_np)
    if mask is None:
        return {"error": "No face detected"}

    rgb, mask = rgb.to(device), mask.to(device)
    inp = torch.cat([rgb, mask], dim=0).unsqueeze(0)

    with torch.no_grad():
        pred = G(inp)[0]

    out_img = (pred.clamp(0,1).cpu().permute(1,2,0).numpy()*255).astype(np.uint8)
    pil_img = Image.fromarray(out_img)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")
