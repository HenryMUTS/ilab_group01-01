import io
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import streamlit as st
from streamlit_image_comparison import image_comparison

# ---------- Page setup ----------
st.set_page_config(page_title="Image Comparison", layout="wide")

st.title("🔍 Image Comparison UI (Input vs. Output)")
st.caption("Upload two images (original + model output) and compare with a draggable slider or blend slider.")

# ---------- Helpers ----------
def load_image(file) -> Image.Image:
    """Load an uploaded file into a PIL Image (RGBA for blending safety)."""
    img = Image.open(file)
    # Convert to RGBA for consistent blending; keep original for the draggable slider
    return img

def ensure_same_size(img_a: Image.Image, img_b: Image.Image) -> tuple[Image.Image, Image.Image]:
    """Resize img_b to match img_a size (keeps it simple and avoids blank renders)."""
    if img_a.size != img_b.size:
        img_b = img_b.resize(img_a.size, Image.LANCZOS)
    return img_a, img_b

def make_demo_image(text: str, size=(800, 500), bg=(240, 240, 240), fg=(30, 30, 30)) -> Image.Image:
    """Generate a demo image with labeled text so the UI never looks blank."""
    img = Image.new("RGB", size, bg)
    d = ImageDraw.Draw(img)
    # Draw some shapes to make the image visibly different
    d.rectangle([60, 60, size[0] - 60, size[1] - 120], outline=fg, width=6)
    d.ellipse([size[0]//2 - 120, size[1]//2 - 120, size[0]//2 + 120, size[1]//2 + 120], outline=fg, width=6)

    # Draw text (no measuring needed)
    d.text((40, size[1] - 80), text, fill=fg)

    return img

def to_rgba(img: Image.Image) -> Image.Image:
    return img.convert("RGBA") if img.mode != "RGBA" else img

# ---------- Sidebar inputs ----------
with st.sidebar:
    st.header("Inputs")
    use_demo = st.toggle("Use demo images", value=True, help="Shows built-in demo images so the page is never blank.")
    st.divider()
    input_file = st.file_uploader("Upload **Input/Original**", type=["png", "jpg", "jpeg", "webp"], key="input_upl")
    output_file = st.file_uploader("Upload **Output/Prediction**", type=["png", "jpg", "jpeg", "webp"], key="output_upl")
    st.caption("Tip: If sizes differ, the app will resize the output to match the input for comparison.")

# ---------- Load images (demo fallback) ----------
if use_demo and not (input_file and output_file):
    img_input = make_demo_image("INPUT / ORIGINAL")
    img_output = make_demo_image("OUTPUT / PREDICTION")
    # Add a visible difference to the output demo
    draw = ImageDraw.Draw(img_output)
    draw.rectangle([120, 100, 360, 220], fill=(255, 255, 255))
else:
    if not input_file or not output_file:
        st.info("👆 Upload both images in the sidebar or enable **Use demo images**.")
        st.stop()
    img_input = load_image(input_file)
    img_output = load_image(output_file)
    record = {
        "user": st.session_state["current_user"],
        "input_image": img_input,
        "output_image": img_output,
    }
    st.session_state.setdefault("records", []).append(record)


# Ensure same size for both comparison modes
img_input, img_output = ensure_same_size(img_input, img_output)

# ---------- Tabs for two comparison modes ----------
tab1, tab2 = st.tabs(["🧲 Before/After (Draggable Slider)", "🎚️ Blend (Opacity Slider)"])

with tab1:
    st.subheader("Before/After")
    st.caption("Drag the vertical handle to reveal the other image.")
    # image_comparison accepts any PIL image (keeps original modes)
    image_comparison(
        img1=img_input,
        img2=img_output,
        label1="Input / Original",
        label2="Output / Prediction",
        show_labels=True,
        make_responsive=True,
        starting_position=50,  # percent
        in_memory=True,
    )

with tab2:
    st.subheader("Blend")
    st.caption("Slide to adjust opacity between the two images (0 = full input, 100 = full output).")

    col_left, col_right = st.columns([1, 3])
    with col_left:
        alpha_pct = st.slider("Blend %", min_value=0, max_value=100, value=50, step=1)
        st.write(f"Alpha: **{alpha_pct/100:.2f}**")

    with col_right:
        a = to_rgba(img_input)
        b = to_rgba(img_output)
        blended = Image.blend(a, b, alpha_pct / 100.0)
        st.image(blended, caption=f"Blended view ({alpha_pct}%)", use_container_width=True)

# ---------- Debug/metadata footer ----------
with st.expander("ℹ️ Details / Debug"):
    st.write(
        {
            "input_size": img_input.size,
            "output_size": img_output.size,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
    )

if "current_user" not in st.session_state:
    st.warning("⚠ Please fill out your details on the Info page first.")

# 🔗 Back link
st.page_link("app.py", label="⬅ Back to Info Page")