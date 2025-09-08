import streamlit as st
import datetime

st.set_page_config(page_title="Image Comparison App", layout="wide")

st.title("👤 User Information")

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
        "Select a date:",
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

            st.session_state.user_submitted = True

            # ✅ Redirect to comparison page
            st.switch_page("pages/main.py")