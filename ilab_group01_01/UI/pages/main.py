import base64
import cv2
import io
from io import BytesIO
import time
import numpy as np
import zipfile
from datetime import datetime
from PIL import Image, ImageDraw
import streamlit as st
import requests
from streamlit_image_comparison import image_comparison
from streamlit_drawable_canvas import st_canvas

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

def img_to_base64(img):
    buffer = BytesIO()
    img.save(buffer, format='PNG')  # Save as PNG (you can change to 'JPEG' if preferred)
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return img_base64

from PIL import Image

def resize_image_to_fit(img, max_width=300, max_height=800):
    """
    Resizes a PIL Image to fit within max_width x max_height while preserving aspect ratio.
    
    Args:
        img (PIL.Image): Original image.
        max_width (int): Maximum width in pixels (default 300 for narrow images).
        max_height (int): Maximum height in pixels (default 800 for tall 1:2.7 ratio).
    
    Returns:
        PIL.Image: Resized image (smaller or same size if already fits).
    """
    # Get original dimensions
    orig_width, orig_height = img.size
    
    # Calculate scaling factors
    scale_width = max_width / orig_width
    scale_height = max_height / orig_height
    
    # Use the smaller scale to fit within both bounds
    scale = min(scale_width, scale_height, 1.0)  # 1.0 ensures no upscaling
    
    if scale < 1.0:
        new_width = int(orig_width * scale)
        new_height = int(orig_height * scale)
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)  # High-quality resize
    
    return img


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
            with zipfile.ZipFile(BytesIO(response.content)) as zip_ref:
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

# --- Initialize canvas data state ---
if "canvas_data" not in st.session_state:
    st.session_state["canvas_data"] = None

if "active_tab" not in st.session_state:
    st.session_state["active_tab"] = "Tab 1"

if "show_canvas" not in st.session_state:
    st.session_state["show_canvas"] = False

# ---------- Tabs ----------
tabs = ["Tab 1", "Tab 2"]
selected_tab = st.radio("Navigation", tabs, index=tabs.index(st.session_state["active_tab"]), horizontal=True)
st.session_state["active_tab"] = selected_tab

# ==================================================
# TAB 1: PRESERVED LOGIC WITH API OUTPUT IMAGE
# ==================================================
if st.session_state["active_tab"] == "Tab 1":
    # Inject CSS for fixed-size bordered image containers (now secondary to resizing)
    st.markdown("""
    <style>
    .fixed-image-container {
        border: 2px solid #ddd;
        border-radius: 8px;
        padding: 10px;
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 10px auto;
        background-color: #f9f9f9;
        max-width: 350px;  /* Slightly wider to accommodate resized width */
        max-height: 850px; /* To fit tall images */
    }
    .fixed-image-container img {
        max-width: 100%;
        max-height: 100%;
        object-fit: contain;
        border-radius: 4px;
    }
    </style>
    """, unsafe_allow_html=True)

    if "active_view" not in st.session_state:
        st.session_state["active_view"] = "Slider"

    active_view = st.selectbox(
        "Select Functionality:",
        ["Slider", "Blend"],
        index=0 if st.session_state["active_view"] == "Slider" else 1
    )

    st.session_state["active_view"] = active_view

    # Resize original images once for consistent sizing across views
    resized_input = resize_image_to_fit(img_input)
    resized_output = resize_image_to_fit(img_output)

    # ---------- Render selected functionality ----------
    if active_view == "Slider":
        st.subheader("Before/After (Slider)")
        
        # Wrap in container; pass resized images
        st.markdown('<div class="fixed-image-container">', unsafe_allow_html=True)
        image_comparison(
            img1=resized_input,
            img2=resized_output,
            label1="Input / Original",
            label2="Output / Prediction",
            show_labels=True,
            make_responsive=False,  # Disable responsive to respect image size
            starting_position=50,
            in_memory=True,
            # If your image_comparison supports 'width', add: width=300
        )
        st.markdown('</div>', unsafe_allow_html=True)

    elif active_view == "Blend":
        st.subheader("Blend (Opacity)")
        
        # Resize after to_rgba (preserves alpha)
        a = resize_image_to_fit(to_rgba(img_input))
        b = resize_image_to_fit(to_rgba(img_output))
        
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
                
                # Use st.image with fixed width for sizing (simpler than base64/HTML)
                placeholder.image(blended, caption=f"Blend α={alpha:.2f}", width=300, clamp=True)
                time.sleep(0.05)
            st.session_state["blend_state"] = target
        else:
            if st.session_state["blend_state"] == "original":
                placeholder.image(a, caption="Original", width=300, clamp=True)
            else:
                placeholder.image(b, caption="Prediction", width=300, clamp=True)

# ==================================================
# TAB 2: NEW MASK OUTPUT VIEW
# ==================================================
elif st.session_state["active_tab"] == "Tab 2":
    st.subheader("🧠 Re-Train (Custom Nose Mask)")
    st.markdown("Draw on the predicted mask below to refine it.")

    # --- Prepare canvas background once ---
    mask_for_canvas = mask_output.convert("RGB")

    canvas_width, canvas_height = mask_for_canvas.width, mask_for_canvas.height

    # "Show Canvas" only works here
    if st.button("🖌️ Show Canvas"):
        st.session_state["show_canvas"] = True

    # Display the canvas if enabled
    if st.session_state["show_canvas"]:
        # (Use your real mask image here)
        mask_img = Image.new("RGB", (400, 400), (255, 255, 255))
        # --- Show stable canvas only when "Re-Train" selected ---
        canvas_result = st_canvas(
            stroke_width=8,
            stroke_color="#FF0000",
            background_image=mask_for_canvas,
            height=canvas_height,
            width=canvas_width,
            drawing_mode="freedraw",
            key="root_canvas",
            update_streamlit=True,
        )

        # --- Persist drawing ---
        if canvas_result.image_data is not None:
            st.session_state["canvas_data"] = canvas_result.image_data

    # --- Manual Update Preview button ---
    st.markdown("### 🖼️ Preview Custom Nose Mask")
    if st.button("🔄 Update Preview"):
        image_data = st.session_state.get("canvas_data", None)

        if image_data is not None and image_data.size > 0:
            mask_array = np.array(image_data)[:, :, 3]
            mask_bw = np.where(mask_array > 50, 255, 0).astype(np.uint8)
            filled_mask = mask_bw.copy()
            contours, _ = cv2.findContours(mask_bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(filled_mask, contours, -1, color=255, thickness=-1)

            custom_mask = Image.fromarray(filled_mask, mode="L")

            overlay = Image.new("RGBA", mask_for_canvas.size, (255, 0, 0, 0))
            overlay_pixels = overlay.load()
            for y in range(filled_mask.shape[0]):
                for x in range(filled_mask.shape[1]):
                    if filled_mask[y, x] > 0:
                        overlay_pixels[x, y] = (255, 0, 0, 128)

            preview_image = Image.alpha_composite(mask_for_canvas.convert("RGBA"), overlay)

            st.session_state["custom_mask"] = custom_mask
            st.session_state["preview_image"] = preview_image

            st.image(custom_mask, caption="Binary Custom Mask (Filled)", use_column_width=True)
            st.image(preview_image, caption="Overlay Preview", use_column_width=True)
        else:
            st.warning("⚠️ Draw something on the canvas first.")

    # --- Re-Train button using latest custom_mask from session_state ---
    if st.button("🚀 Generate & Re-Train"):
        custom_mask = st.session_state.get("custom_mask", None)

        # Ensure both have the same size
        img_input_resized, custom_mask_resized = ensure_same_size(img_input, custom_mask)

        if img_input_resized is None:
            st.warning("⚠️ Please upload a face image first.")
        else:
            # --- Convert PIL images to BytesIO before sending ---
            def pil_to_bytesio(pil_img, format="PNG"):
                buf = io.BytesIO()
                pil_img.save(buf, format=format)
                buf.seek(0)
                return buf

            input_buf = pil_to_bytesio(img_input_resized)
            mask_buf = pil_to_bytesio(custom_mask_resized) if custom_mask_resized else None

            with st.spinner("Sending image to API..."):
                try:
                    if mask_buf:
                        files = {
                            "file": ("input.png", input_buf, "image/png"),
                            "mask_file": ("mask.png", mask_buf, "image/png"),
                        }
                    else:
                        files = {
                            "file": ("input.png", input_buf, "image/png"),
                        }

                    # Send request
                    response = requests.post(API_URL, files=files)

                    if response.status_code == 200:
                        zip_bytes = io.BytesIO(response.content)
                        with zipfile.ZipFile(zip_bytes, "r") as z:
                            pred_img = Image.open(z.open("prediction.png"))
                            mask_overlay = Image.open(z.open("mask_overlay.png"))

                        st.success("✅ Prediction completed successfully.")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.image(img_input_resized, caption="Input Image", use_column_width=True)
                        with col2:
                            st.image(pred_img, caption="Predicted Output", use_column_width=True)
                        with col3:
                            st.image(mask_overlay, caption="Mask Overlay", use_column_width=True)

                        st.download_button(
                            label="⬇️ Download Results ZIP",
                            data=response.content,
                            file_name="results.zip",
                            mime="application/zip",
                        )
                    else:
                        st.error(f"❌ API error: {response.status_code} — {response.text}")
                except Exception as e:
                    st.error(f"🚨 Request failed: {e}")


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