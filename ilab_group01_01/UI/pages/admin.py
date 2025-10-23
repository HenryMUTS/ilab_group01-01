import streamlit as st
import os
from datetime import datetime
from PIL import Image
import io
import sys

# Add the root directory to Python path to import st_theme
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.insert(0, root_dir)
from st_theme import set_page_theme, show_callout

# ---------- Page setup ----------
set_page_theme(title="NoseVision AI", icon="admin_panel_settings")

st.caption("Admin Page - Manage User Submissions")

records = st.session_state.get("records", [])
if not records:
    st.info("No submissions available yet.")
    st.stop()

# Show only 20 most recent
records_to_show = records[-20:]

selected = []
with st.form("admin_form"):
    for idx, rec in enumerate(records_to_show):
        st.markdown("---")  # divider between records
        with st.container():
            c1, c2 = st.columns([2, 3])  # 2 for images, 3 for text
            with c1:
                st.image(rec["input_image"], caption=f"Input #{idx}", width=200)
                st.image(rec["output_image"], caption=f"Output #{idx}", width=200)
            with c2:
                user = rec.get("user", {})
                st.markdown(
                    f"""
                    <div style="padding-left:40px;">
                    <h4>User Info</h4>
                    <b>First Name:</b> {user.get("fname", "")}<br>
                    <b>Last Name:</b> {user.get("lname", "")}<br>
                    <b>Gender:</b> {user.get("gender", "")}<br>
                    <b>Date of Birth:</b> {user.get("DOB", "")}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            approved = st.checkbox("Approve", key=f"approve_{idx}")
            if approved:
                selected.append(rec)


    submitted = st.form_submit_button("Save Selected")

if submitted and selected:
    save_dir = "ilab_group01_01/UI/saved_records"
    os.makedirs(save_dir, exist_ok=True)

    details_path = os.path.join(save_dir, "details.txt")
    with open(details_path, "a") as f:
        for rec in selected:
            user = rec["user"]

            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            base_name = f"{user.get('fname','unknown')}_{user.get('lname','')}_{timestamp}"

            input_filename = f"{base_name}_input.png"
            output_filename = f"{base_name}_output.png"

            def save_image(data, path):
                if isinstance(data, bytes):
                    img = Image.open(io.BytesIO(data))
                elif isinstance(data, Image.Image):
                    img = data
                else:
                    raise TypeError(f"Unsupported type for image saving: {type(data)}")
                img.save(path)

            input_path = os.path.join(save_dir, input_filename)
            output_path = os.path.join(save_dir, output_filename)

            save_image(rec["input_image"], input_path)
            save_image(rec["output_image"], output_path)

            f.write(f"First Name: {user.get('fname','')}\n")
            f.write(f"Last Name: {user.get('lname','')}\n")
            f.write(f"Gender: {user.get('gender','')}\n")
            f.write(f"Date of Birth: {user.get('DOB','')}\n")
            f.write(f"Input Image: {input_filename}\n")
            f.write(f"Output Image: {output_filename}\n")
            f.write("---\n")

    st.success(f"Saved {len(selected)} submissions to {save_dir}")
elif submitted:
    st.warning("No submissions selected.")
