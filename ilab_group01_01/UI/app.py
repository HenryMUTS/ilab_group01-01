import streamlit as st
import datetime
import sys
import os

# Add the root directory to Python path to import st_theme
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, root_dir)
from st_theme import set_page_theme, show_callout

# ---------- Page setup ----------
set_page_theme(title="NoseVision AI", icon="person")

st.caption("Enter Profile Information To Continue..")

# Session state setup
if "user_submitted" not in st.session_state:
    st.session_state.user_submitted = False

today = datetime.date.today()

with st.form("user_form"):
    # name = st.text_input("Full Name *")
    first_name = st.text_input("First Name *")
    last_name = st.text_input("Last Name *")
    gender = st.selectbox("Gender *", ["", "Male", "Femal", "Others"])
    dob = st.date_input(
        "Date of Birth:",
        value=today,  # default value
        min_value=today - datetime.timedelta(days=25620),  # earliest selectable date
        max_value=today,  # latest selectable date
        format="YYYY.MM.DD"
    )
    # email = st.text_input("Email Address *")
    # role = st.selectbox("Role *", ["", "Researcher", "Student", "Clinician", "Other"])
    # agree = st.checkbox("I agree to the terms and conditions *")

    submitted = st.form_submit_button("Continue")

    if submitted:
        if not first_name.strip() or last_name == "" or not gender or not dob:
            st.error("⚠️ Please complete **all required fields** and agree to continue.")
        else:
            st.session_state["current_user"] = {
                "fname": first_name.strip(),
                "lname": last_name.strip(),
                "gender": gender,
                "DOB": dob
                # "email": email.strip(),
                # "role": role,
            }
            st.success("✅ Details saved. You can now proceed to the comparison page.")
            
            # ✅ Redirect to comparison page
            st.switch_page("pages/main.py")