import io
import time
import numpy as np
import zipfile
from datetime import datetime
from PIL import Image, ImageDraw
import streamlit as st
import requests
from streamlit_image_comparison import image_comparison

# ---------- Page setup ----------
st.set_page_config(page_title="Image Comparison", layout="wide")

st.title("🔍 Image Comparison UI (Input vs. Prediction)")
st.caption("Upload an image, run the model, and compare input vs. prediction with draggable/blend sliders.")

API_URL = "http://127.0.0.1:8000/predict"  # FastAPI inference endpoint

# ---------- Helpers ----------
def load_image(file) -> Image.Image:
    file.seek(0)
    return Image.open(file)

def ensure_same_size(img_a: Image.Image, img_b: Image.Image):
    if img_a.size != img_b.size:
        img_b = img_b.resize(img_a.size, Image.LANCZOS)
    return img_a, img_b

def make_demo_image(text: str, size=(800, 500), bg=(240, 240, 240), fg=(30, 30, 30)):
    img = Image.new("RGB", size, bg)
    d = ImageDraw.Draw(img)
    d.rectangle([60, 60, size[0] - 60, size[1] - 120], outline=fg, width=6)
    d.text((40, size[1] - 80), text, fill=fg)
    return img

def to_rgba(img: Image.Image) -> Image.Image:
    return img.convert("RGBA") if img.mode != "RGBA" else img

def get_prediction(input_file):
    try:
        input_file.seek(0)
        files = {"file": (input_file.name, input_file.read(), input_file.type)}
    except Exception:
        st.error("⚠️ Failed to read uploaded file.")
        return None, None

    with st.spinner("⏳ Running prediction... please wait..."):
        try:
            response = requests.post(API_URL, files=files, timeout=120)
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Connection error: {e}")
            return None, None

    # Expect a ZIP file with "prediction" and "mask"
    if response.status_code == 200 and response.headers.get("content-type", "").startswith("application/zip"):
        try:
            with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
                pred_img = None
                mask_img = None
                for name in zip_ref.namelist():
                    with zip_ref.open(name) as f:
                        img = Image.open(f).convert("RGB")
                        if "pred" in name.lower():
                            pred_img = img
                        elif "mask" in name.lower():
                            mask_img = img
                if pred_img is None:
                    st.error("❌ No prediction image found in ZIP.")
                return pred_img, mask_img
        except Exception as e:
            st.error(f"⚠️ Error reading ZIP: {e}")
            return None, None

    st.error("❌ Unexpected API response (expected ZIP).")
    return None, None

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Inputs")
    use_demo = st.toggle("Use demo image", value=True)
    st.divider()
    input_file = st.file_uploader(
        "Upload **Input/Original**",
        type=["png", "jpg", "jpeg", "webp"],
        key="input_upl"
    )
    run_pred = st.button("Run Prediction")
    st.caption("Tip: Model output will be generated via API.")

# ---------- Handle missing user info ----------
if "current_user" not in st.session_state:
    st.warning("⚠ Please fill out your details on the Info page first.")
    st.page_link("./app.py", label="⬅ Go to Info Page")
    st.stop()

# ---------- Load or create record store ----------
if "records" not in st.session_state:
    st.session_state["records"] = []

# ---------- Load input/output images ----------
if use_demo and not input_file:
    img_input = make_demo_image("INPUT / ORIGINAL")
    img_output = make_demo_image("OUTPUT / PREDICTION")
else:
    if not input_file:
        st.info("👆 Upload an image in the sidebar or enable **Use demo image**.")
        st.stop()

    img_input = load_image(input_file)

    # Reset prediction when filename changes
    if "last_filename" not in st.session_state or st.session_state["last_filename"] != input_file.name:
        st.session_state["last_prediction"] = None
        st.session_state["last_filename"] = input_file.name

    img_output = st.session_state.get("last_prediction")

if run_pred:
    pred_img, mask_img = get_prediction(input_file)
    if pred_img:  # ✅ use pred_img, not img_output
        # Save prediction and ensure consistent size
        img_input = Image.open(input_file).convert("RGB")
        img_input, pred_img = ensure_same_size(img_input, pred_img)
        if mask_img:
            _, mask_img = ensure_same_size(img_input, mask_img)

        # ✅ Store results in session_state so they persist
        st.session_state["img_input"] = img_input
        st.session_state["img_output"] = pred_img
        st.session_state["mask_output"] = mask_img

        record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "user": st.session_state.get("current_user", {}),
            "input_image": img_input,
            "output_image": pred_img,
            "filename": input_file.name,
        }
        st.session_state["records"].append(record)
    else:
        st.error("⚠️ Prediction failed or invalid API response.")

    if pred_img is None:
        st.warning("⚠️ No prediction yet. Click **Run Prediction** to generate output.")

# ---------- Match sizes ----------
img_input = st.session_state.get("img_input")
img_output = st.session_state.get("img_output")
mask_output = st.session_state.get("mask_output")

if not img_input:
    st.info("👆 Upload an image and click **Run Prediction** to start.")
    st.stop()

# ---------- Tabs ----------
tab1, tab2 = st.tabs(["🎨 Original vs Prediction", "🧠 Re-Train"])

# ==================================================
# TAB 1: PRESERVED LOGIC WITH API OUTPUT IMAGE
# ==================================================
with tab1:
    if "active_view" not in st.session_state:
        st.session_state["active_view"] = "Slider"

    active_view = st.selectbox(
        "Select Functionality:",
        ["Slider", "Blend"],
        index=0 if st.session_state["active_view"] == "Slider" else 1
    )

    st.session_state["active_view"] = active_view

    # ---------- Render selected functionality ----------
    if active_view == "Slider":
        st.subheader("Before/After (Slider)")
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

    elif active_view == "Blend":
        st.subheader("Blend (Opacity)")
        a, b = to_rgba(img_input), to_rgba(img_output)
        placeholder = st.empty()

        # Initialize toggle state if not exists
        if "blend_state" not in st.session_state:
            st.session_state["blend_state"] = "original"

        # Blend toggle button
        toggle = st.button("Switch Original/Prediction", key="blend_btn")

        # Animate gradual blend on button press
        if toggle:
            target = "prediction" if st.session_state["blend_state"] == "original" else "original"
            alphas = np.linspace(0, 1, 25) if target == "prediction" else np.linspace(1, 0, 25)
            for alpha in alphas:
                blended = Image.blend(a, b, alpha)
                placeholder.image(blended, caption=f"Blend α={alpha:.2f}", use_container_width=True)
                time.sleep(0.05)
            st.session_state["blend_state"] = target
        else:
            if st.session_state["blend_state"] == "original":
                placeholder.image(a, caption="Original", use_container_width=True)
            else:
                placeholder.image(b, caption="Prediction", use_container_width=True)


# ==================================================
# TAB 2: NEW MASK OUTPUT VIEW
# ==================================================
with tab2:
    st.subheader("Re-Train (Mask Output)")
    if mask_output:
        # Ensure same size with input before showing
        _, mask_output = ensure_same_size(img_input, mask_output)
        st.image(mask_output, caption="Mask Output", use_container_width=True)
    else:
        st.info("⚡ Run prediction first to view mask output.")


# ---------- Debug ----------
with st.expander("ℹ️ Details / Debug"):
    st.write({
        "input_size": img_input.size,
        "output_size": img_output.size,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "filename": getattr(input_file, "name", None),
        "record_count": len(st.session_state["records"]),
    })

# st.page_link("./pages/admin.py", label="➡ Go to Results Page")