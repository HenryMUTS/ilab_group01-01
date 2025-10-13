import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image

st.title("Canvas Test")
test_img = Image.new("RGB", (400, 400), (128, 128, 128))  # Gray background
bg_array = test_img
canvas = st_canvas(background_image=bg_array, height=400, width=400, stroke_color="red", stroke_width=10, drawing_mode="freedraw")
if canvas.image_data is not None:
    alpha = np.array(canvas.image_data)[:, :, 3]
    st.image(Image.fromarray(np.where(alpha > 30, 255, 0).astype(np.uint8), "L"), caption="Drawn Mask")