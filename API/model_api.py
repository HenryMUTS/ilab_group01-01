import io
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from PIL import Image
import numpy as np
import face_alignment
from skimage.draw import polygon

app = FastAPI()
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device="cpu", flip_input=False)


# -------------------------------
# 1. Define / import your model
# -------------------------------
# Replace this with your real model architecture
# ----- tiny mask helpers
def get_nose_mask(img_pil, size=(224, 224)):
    """
    Takes a PIL image, detects face landmarks, and returns a binary mask of the nose.
    """
    img_np = np.array(img_pil)  # RGB numpy image
    preds = fa.get_landmarks(img_np)

    if preds is None:
        # If no face detected, return an all-zeros mask
        return np.zeros(size, dtype=np.float32)

    # Take first face only
    landmarks = preds[0]

    # Nose landmarks in 68-point model = [27–35]
    nose_points = landmarks[27:36]  

    # Resize landmarks to match target mask size
    h, w = img_np.shape[:2]
    scale_x = size[0] / w
    scale_y = size[1] / h
    nose_points_scaled = np.array([
        [x * scale_x, y * scale_y] for (x, y) in nose_points
    ])

    # Create mask
    mask = np.zeros(size, dtype=np.float32)
    rr, cc = polygon(nose_points_scaled[:, 1], nose_points_scaled[:, 0], mask.shape)
    mask[rr, cc] = 1.0
    return mask


def dilate_mask_binary(mask, k=11, iters=2):
    out = mask
    for _ in range(iters):
        out = F.max_pool2d(out, kernel_size=k, stride=1, padding=k//2)
    return out.clamp(0,1)

def feather_mask(mask, k=9):
    return F.avg_pool2d(mask, kernel_size=k, stride=1, padding=k//2).clamp(0,1)

def match_input_size(input_img: Image.Image, output_img: Image.Image) -> Image.Image:
    """
    Ensure output_img matches the size of input_img.
    Keeps original aspect ratio of input.
    """
    if input_img.size != output_img.size:
        output_img = output_img.resize(input_img.size, Image.LANCZOS)
    return output_img


# --------------------------- UNet blocks ---------------------------
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
        # pad if needed (in case odd dims)
        diffY = skip.size(2) - x.size(2)
        diffX = skip.size(3) - x.size(3)
        if diffY != 0 or diffX != 0:
            x = F.pad(x, (0, diffX, 0, diffY))
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


# --------------------------- UNet Generator (residual) ---------------------------
class UNetNoseGenerator(nn.Module):
    """
    UNet that takes RGB+mask (4ch) and predicts a 3ch residual 'delta'.
    Output = rgb + soft_mask * tanh(raw) * scale
    """
    def __init__(self, in_ch=4, out_ch=3, base=64, depth=5, res_max=0.75):
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
        scale = torch.sigmoid(self._alpha) * self.res_max
        delta = torch.tanh(raw) * scale     # bounded residual

        # soft blend within a feathered mask band
        hard   = dilate_mask_binary(mask1, k=11, iters=2)
        m_soft = feather_mask(hard, k=9)            # [B,1,H,W]
        m3     = m_soft.repeat(1, 3, 1, 1)
        full_rgb = (rgb + delta)
        out    = rgb + delta * m3
        if return_full:
            return out, full_rgb, m_soft
        return out


# -------------------------------
# 2. Load checkpoint
# -------------------------------
checkpoint_path = "C:/Users/henry/OneDrive/Desktop/UTS/iLab/Model/Practical/Niketh_Unet_PatchGan_Model.pt"

checkpoint = torch.load(checkpoint_path, map_location="cpu")

# Some GAN checkpoints store generator under "G"
if "G" in checkpoint:
    state_dict = checkpoint["G"]
elif "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
else:
    # fallback: assume the checkpoint is already the state_dict
    state_dict = checkpoint

# Load into your generator
model = UNetNoseGenerator()
model.load_state_dict(state_dict, strict=False)  # strict=False ignores non-matching keys
model.eval()

# -------------------------------
# 3. Preprocess / Postprocess
# -------------------------------
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),   # adjust to your model’s input
    transforms.ToTensor(),
])

to_pil = transforms.ToPILImage()

# -------------------------------
# 4. API Endpoint
# -------------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")

    # Resize for model
    img_resized = img.resize((224, 224))

    # Get RGB tensor
    rgb_tensor = transform(img_resized)  # [3,H,W]

    # Get nose mask
    mask_np = get_nose_mask(img, size=(224, 224))
    mask_tensor = torch.tensor(mask_np).unsqueeze(0)  # [1,H,W]

    # Combine RGB + mask
    x = torch.cat([rgb_tensor, mask_tensor], dim=0).unsqueeze(0)  # [1,4,H,W]

    with torch.no_grad():
        y = model(x)

    out_img = to_pil(y.squeeze())

    out_img = match_input_size(img, out_img)

    buf = io.BytesIO()
    out_img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")