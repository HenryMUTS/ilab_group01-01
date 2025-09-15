import io
import time
import numpy as np
from datetime import datetime
from PIL import Image, ImageDraw
import streamlit as st
import requests
from streamlit_image_comparison import image_comparison

# ---------- Page setup ----------
st.set_page_config(page_title="Image Comparison", layout="wide")

st.title("🔍 Image Comparison UI (Input vs. Prediction)")
st.caption("Upload an image, run the model, and compare input vs. prediction with draggable/blend sliders.")

API_URL = "http://127.0.0.1:8000/predict"  # <-- update if your FastAPI runs elsewhere

# ---------- Helpers ----------
def load_image(file) -> Image.Image:
    """Load an uploaded file into a PIL Image (RGBA for blending safety)."""
    return Image.open(file)

def ensure_same_size(img_a: Image.Image, img_b: Image.Image) -> tuple[Image.Image, Image.Image]:
    """Resize img_b to match img_a size (keeps it simple and avoids blank renders)."""
    if img_a.size != img_b.size:
        img_b = img_b.resize(img_a.size, Image.LANCZOS)
    return img_a, img_b

def make_demo_image(text: str, size=(800, 500), bg=(240, 240, 240), fg=(30, 30, 30)) -> Image.Image:
    """Generate a demo image with labeled text so the UI never looks blank."""
    img = Image.new("RGB", size, bg)
    d = ImageDraw.Draw(img)
    d.rectangle([60, 60, size[0] - 60, size[1] - 120], outline=fg, width=6)
    d.ellipse([size[0]//2 - 120, size[1]//2 - 120, size[0]//2 + 120, size[1]//2 + 120], outline=fg, width=6)
    d.text((40, size[1] - 80), text, fill=fg)
    return img

def to_rgba(img: Image.Image) -> Image.Image:
    return img.convert("RGBA") if img.mode != "RGBA" else img

def get_prediction(input_file) -> Image.Image:
    """Send input image to API and get prediction back."""
    files = {"file": input_file.getvalue()}
    response = requests.post(API_URL, files=files)
    if response.status_code == 200:
        return Image.open(io.BytesIO(response.content))
    else:
        st.error(f"API Error {response.status_code}: {response.text}")
        return None

def pil_to_bytes(img: Image.Image) -> bytes:
    """Convert PIL image to bytes (PNG format)."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------- Sidebar inputs ----------
with st.sidebar:
    st.header("Inputs")
    use_demo = st.toggle("Use demo image", value=True, help="Shows a built-in demo if no upload.")
    st.divider()
    input_file = st.file_uploader("Upload **Input/Original**", type=["png", "jpg", "jpeg", "webp"], key="input_upl")
    run_pred = st.button("Run Prediction")
    st.caption("Tip: Model output will be generated via API.")

# ---------- Load images (demo fallback) ----------
if use_demo and not input_file:
    img_input = make_demo_image("INPUT / ORIGINAL")
    img_output = make_demo_image("OUTPUT / PREDICTION")
    draw = ImageDraw.Draw(img_output)
    draw.rectangle([120, 100, 360, 220], fill=(255, 255, 255))
else:
    if not input_file:
        st.info("👆 Upload an image in the sidebar or enable **Use demo image**.")
        st.stop()

    img_input = load_image(input_file)
    img_output = st.session_state.get("last_prediction")

    if run_pred:
        img_output = get_prediction(input_file)
        if img_output:
            st.session_state["last_prediction"] = img_output
            record = {
                "user": st.session_state.get("current_user", {}),
                "input_image": img_input,
                "output_image": img_output,
            }
            st.session_state.setdefault("records", []).append(record)

    if img_output is None:
        st.info("⚡ Upload an image and click **Run Prediction** to see results.")
        st.stop()

# Ensure same size
img_input, img_output = ensure_same_size(img_input, img_output)

# ---------- Tabs for two comparison modes ----------
tab1, tab2 = st.tabs(["🧲 Before/After (Slider)", "🎚️ Blend (Opacity)"])

with tab1:
    st.subheader("Before/After")
    image_comparison(
        img1=img_input,
        img2=img_output,
        label1="Input / Original",
        label2="Output / Prediction",
        show_labels=True,
        make_responsive=True,
        starting_position=50,
        in_memory=True,
    )

with tab2:
    st.subheader("Blend")
    st.caption("Automatic animation: image opacity changes in real time.")

    a = to_rgba(img_input)
    b = to_rgba(img_output)

    # Animation controls
    rerun = st.button("Cycle Animation")

    placeholder = st.empty()

    if rerun:
        # Forward fade (0 → 1)
        for alpha in np.linspace(0, 1, 15):
            blended = Image.blend(a, b, alpha)
            placeholder.image(blended, caption=f"Blended view ({alpha:.2f})", width='stretch')
            time.sleep(0.05)

        # Backward fade (1 → 0)
        for alpha in np.linspace(1, 0, 15):
            blended = Image.blend(a, b, alpha)
            placeholder.image(blended, caption=f"Blended view ({alpha:.2f})", width='stretch')
            time.sleep(0.05)

# ---------- Debug footer ----------
with st.expander("ℹ️ Details / Debug"):
    st.write({
        "input_size": img_input.size,
        "output_size": img_output.size,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    })

if "current_user" not in st.session_state:
    st.warning("⚠ Please fill out your details on the Info page first.")

# 🔗 Back link
st.page_link("app.py", label="⬅ Back to Info Page")