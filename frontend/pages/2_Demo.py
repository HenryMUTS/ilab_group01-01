# --- Demo page: no backend required ---
import io, base64, re, os
from pathlib import Path
from PIL import Image
import streamlit as st

st.set_page_config(page_title="Demo — Rhinoplasty Predictor", layout="centered")

# Mock assets
BASE_DIR = Path(__file__).resolve().parents[1]  # go up from pages/ to frontend/
MOCK_DIR = BASE_DIR / "assets" / "images"
MOCK_INPUT = MOCK_DIR / "mock_input.jpeg"
MOCK_TARGET = MOCK_DIR / "mock_target.jpeg"

# set a fake user for the demo
if "auth" not in st.session_state:
    st.session_state.auth = {"token": None, "role": None, "user": None}
if not st.session_state.auth["token"]:
    st.session_state.auth = {
        "token": "DEMO",
        "role": "customer",
        "user": {"username": "demo_user", "email": "demo@example.com", "role": "customer"},
    }

def _pil_to_data_url(img: Image.Image) -> str:
    buf = io.BytesIO(); img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

def image_compare(orig_img: Image.Image, out_img: Image.Image):
    try:
        from streamlit_image_comparison import image_comparison
        image_comparison(
            img1=orig_img, img2=out_img,
            label1="Input / Original", label2="Output / Prediction",
            show_labels=True, make_responsive=True, starting_position=50, in_memory=True
        )
    except Exception:
        # Fallback: HTML-based slider, no extra package needed
        left = _pil_to_data_url(orig_img); right = _pil_to_data_url(out_img)
        from streamlit.components.v1 import html
        html(f"""
        <link rel="stylesheet" href="https://unpkg.com/img-comparison-slider@7/dist/styles.css">
        <script type="module" src="https://unpkg.com/img-comparison-slider@7/dist/index.js"></script>
        <img-comparison-slider style="width:100%;height:auto;">
          <img slot="first"  style="display:block;max-width:100%;" src="{left}" />
          <img slot="second" style="display:block;max-width:100%;" src="{right}" />
        </img-comparison-slider>
        """, height=520)

st.markdown("""
<style>
  .card{max-width:900px;margin:0 auto;background:#fff;border-radius:12px;
        box-shadow:0 6px 24px rgba(0,0,0,.08);padding:24px;border:1px solid rgba(0,0,0,.06);}
  .center{text-align:center}.title{font-size:20px;font-weight:700;padding:8px 0}
</style>
""", unsafe_allow_html=True)

#st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="center title">Demo — Rhinoplasty Outcome Predictor</div>', unsafe_allow_html=True)
st.caption("This page uses local mock images; no data is sent to any server.")

# Load mock images
if not (MOCK_INPUT.exists() and MOCK_TARGET.exists()):
    st.error("Mock images not found. Place files at `frontend/assets/images/mock_input.jpeg` and `mock_target.jpeg`.")
else:
    img_input = Image.open(MOCK_INPUT).convert("RGB")
    img_output = Image.open(MOCK_TARGET).convert("RGB").resize(img_input.size)

    # Fake metrics
    st.subheader("Results (demo)")
    st.json({"psnr": 31.2, "lpips": 0.21, "model_version": "demo-0.1"})

    tab1, tab2 = st.tabs(["🧲 Before/After (Slider)", "🎚️ Blend (Opacity)"])
    with tab1:
        image_compare(img_input, img_output)
    with tab2:
        alpha = st.slider("Blend % (demo)", 0, 100, 50, 1, key="blend_slider_demo_page")
        blended = Image.blend(img_input.convert("RGBA"), img_output.convert("RGBA"), alpha/100.0)
        st.image(blended, caption=f"Blended view ({alpha}%)", use_container_width=True)

    # Download predicted
    uname = (st.session_state.auth.get("user") or {}).get("username", "user")
    uname_safe = re.sub(r"[^A-Za-z0-9_-]+", "", uname).lower() or "user"
    buf = io.BytesIO(); img_output.save(buf, format="PNG"); buf.seek(0)
    st.download_button("Download Predicted PNG",
        data=buf.getvalue(), file_name=f"predicted_{uname_safe}.png", mime="image/png",
        key="download_pred_demo_page")

# Simple way back to main page
try:
    # Streamlit >= 1.31
    st.page_link("streamlit_app.py", label="Back to Login / Main", icon="↩️")
except Exception:
    st.write("Use the sidebar to navigate back to the main page.")
st.markdown("</div>", unsafe_allow_html=True)
