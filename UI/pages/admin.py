import streamlit as st
import os
from datetime import datetime
from PIL import Image
import io

st.title("Admin Page - Manage User Submissions")

# Ensure we have records
records = st.session_state.get("records", [])
if not records:
    st.info("No submissions available yet.")
    st.stop()

# Show only 20 most recent
records_to_show = records[-20:]

# Checkboxes for selection
selected_indices = []
with st.form("admin_form"):
    st.subheader("Select submissions to save")
    for idx, rec in enumerate(records_to_show):
        col1, col2, col3 = st.columns([1, 2, 2])
        with col1:
            selected = st.checkbox("Select", key=f"sel_{idx}")
        with col2:
            st.write(rec["user"])
            st.image(rec["input_image"], caption="Input Image", use_container_width=True)
        with col3:
            st.image(rec["output_image"], caption="Output Image", use_container_width=True)

        if selected:
            selected_indices.append(idx)

    # Save button
    submitted = st.form_submit_button("💾 Save Selected")

if submitted and selected_indices:
    save_dir = "saved_records"
    os.makedirs(save_dir, exist_ok=True)

    details_path = os.path.join(save_dir, "details.txt")
    with open(details_path, "a") as f:
        for idx in selected_indices:
            rec = records_to_show[idx]
            user = rec["user"]

            # Unique filenames with timestamp
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            base_name = f"{user['fname']}{user['lname']}_{timestamp}"

            input_filename = f"{base_name}_input.png"
            output_filename = f"{base_name}_output.png"

            # Save images
            def save_image(data, path):
                """Save image whether it's bytes or PIL.Image"""
                if isinstance(data, bytes):
                    img = Image.open(io.BytesIO(data))
                elif isinstance(data, Image.Image):  # Already a PIL image
                    img = data
                else:
                    raise TypeError(f"Unsupported type for image saving: {type(data)}")
                img.save(path)
            
            input_path = os.path.join(save_dir, input_filename)
            output_path = os.path.join(save_dir, output_filename)

            save_image(rec["input_image"], input_path)
            save_image(rec["output_image"], output_path)


            # Log details
            f.write(f"First Name: {user['fname']}\n")
            f.write(f"Last Name: {user['lname']}\n")
            f.write(f"Gender: {user['gender']}\n")
            f.write(f"Date of Birth: {user['DOB']}\n")
            # f.write(f"Email: {user['email']}\n")
            # f.write(f"Role: {user['role']}\n")
            f.write(f"Input Image: {input_filename}\n")
            f.write(f"Output Image: {output_filename}\n")
            f.write("---\n")

    st.success(f"Saved {len(selected_indices)} submissions to {save_dir}")
elif submitted:
    st.warning("No submissions selected.")
