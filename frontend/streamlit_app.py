import os, io, requests, re
from pathlib import Path
from datetime import date
from PIL import Image
import streamlit as st


API = os.getenv("API_BASE", "http://localhost:8000/api")

# ---- Mock assets ----
BASE_DIR = Path(__file__).resolve().parent
MOCK_DIR = BASE_DIR / "assets" / "images"
MOCK_INPUT = MOCK_DIR / "mock_input.jpeg"
MOCK_TARGET = MOCK_DIR / "mock_target.jpeg"

# ---- Session bootstrap ----
if "auth" not in st.session_state:
    st.session_state.auth = {"token": None, "role": None, "user": None}
if "route" not in st.session_state:
    st.session_state.route = "login"  # "login" | "signup"

# ---- CSS ----
st.markdown("""
<style>
  .center { text-align:center; }
  .title { font-size:20px; font-weight:700; padding: 12px 0; }
  .subtitle { color:#666; font-size:13px; margin-top:8px; }
  .link-like { color:#2563eb; text-decoration:underline; cursor:pointer; }
</style>
""", unsafe_allow_html=True)

# ---- Title banner ----
st.markdown('<div class="center title">Rhinoplasty Outcome Predictor</div>', unsafe_allow_html=True)


def show_login():
    # allow prefill after signup
    if "prefill_username" in st.session_state and "login_username" not in st.session_state:
        st.session_state["login_username"] = st.session_state.pop("prefill_username")

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="center" style="font-size:18px; font-weight:600;">Login</div>', unsafe_allow_html=True)
    with st.form("login_form"):
        u = st.text_input("Username", key="login_username")
        p = st.text_input("Password", type="password", key="login_password")
        submitted = st.form_submit_button("Login")

    if submitted:
        if not u or not p:
            st.error("Please enter both username and password.")
        else:
            try:
                r = requests.post(f"{API}/auth/token/", json={"username": u, "password": p}, timeout=15)
                if r.status_code == 200:
                    token = r.json()["access"]
                    me = requests.get(f"{API}/me/", headers={"Authorization": f"Bearer {token}"}, timeout=15)
                    if me.ok:
                        st.session_state.auth.update(
                            {"token": token, "user": me.json(), "role": me.json().get("role")}
                        )
                        st.success("Logged in. Loading…")
                        st.rerun()
                    else:
                        st.error("Login ok, but fetching user failed.")
                else:
                    st.error("Invalid username or password.")
            except Exception as e:
                st.error(f"Login error: {e}")

    # link to sign-up
    cols = st.columns([1, 1, 1])
    with cols[1]:
        if st.button("No account? Sign up", key="to_signup"):
            st.session_state.route = "signup"
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


def show_signup():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="center" style="font-size:18px; font-weight:600;">Create an account</div>', unsafe_allow_html=True)

    with st.form("signup_form"):
        su_username = st.text_input("Username", key="signup_username")
        su_email    = st.text_input("Email", key="signup_email")
        su_password = st.text_input("Password", type="password", key="signup_password")
        su_dob      = st.date_input("Date of birth", value=date(2000,1,1), key="signup_dob")
        su_phone    = st.text_input("Phone (optional)", key="signup_phone")
        su_gender   = st.selectbox("Gender", ["female", "male", "unspecified"], key="signup_gender")
        su_med      = st.text_area("Medical conditions (optional)", key="signup_med")
        su_allergy  = st.text_area("Drug allergies (optional)", key="signup_allergy")
        submitted   = st.form_submit_button("Create account")

    if submitted:
        if not su_username or not su_email or not su_password:
            st.error("Username, email, and password are required.")
        else:
            payload = {
                "username": su_username, "email": su_email, "password": su_password,
                "dob": su_dob.isoformat(), "phone": su_phone, "gender": su_gender,
                "medical_conditions": su_med, "drug_allergies": su_allergy,
            }
            try:
                r = requests.post(f"{API}/auth/signup/", json=payload, timeout=20)
                if r.status_code == 201:
                    st.success("Account created. Returning to login…")
                    # prefill username on login and switch route
                    st.session_state["prefill_username"] = su_username
                    st.session_state.route = "login"
                    st.rerun()
                else:
                    st.error(f"Sign-up failed: {r.text}")
            except Exception as e:
                st.error(f"Sign-up error: {e}")

    # link back to login
    cols = st.columns([1, 1, 1])
    with cols[1]:
        if st.button("Back to Login", key="to_login"):
            st.session_state.route = "login"
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


# ---- Render the auth screen until logged in ----
if not st.session_state.auth["token"]:
    if st.session_state.route == "signup":
        show_signup()
    else:
        show_login()
    st.stop()

# ====== From here down: the rest of your app (only shows after login) ======
headers = {"Authorization": f"Bearer {st.session_state.auth['token']}"}
st.set_page_config(page_title='Rhinoplasty Predictor', layout='wide')
# session bootstrap for work state
if "work" not in st.session_state:
    st.session_state.work = {"photo_id": None, "pred": None}

top_l, top_r = st.columns([1, 0.2])
with top_l:
    st.markdown(f"**Signed in as:** {st.session_state.auth['user'].get('username','')}")
with top_r:
    if st.button("Log out", key="header_logout", use_container_width=True):
        # clear session auth + work state
        st.session_state.auth = {"token": None, "role": None, "user": None}
        st.session_state.work = {"photo_id": None, "pred": None}
        st.session_state.route = "login"  # optional, to show login explicitly
        st.rerun()


# handy: username suffix for downloads
_user = (st.session_state.auth.get("user") or {})
_uname = _user.get("username", "user")
_uname_safe = re.sub(r"[^A-Za-z0-9_-]+", "", _uname).lower() or "user"

# --- small util: HTML before/after slider (fallback if component not installed) ---
def _pil_to_data_url(img: Image.Image) -> str:
    b = io.BytesIO(); img.save(b, format="PNG")
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()

def image_compare(orig_img: Image.Image, out_img: Image.Image):
    # Try streamlit-image-comparison if available; else embed HTML slider
    try:
        from streamlit_image_comparison import image_comparison
        image_comparison(
            img1=orig_img, img2=out_img,
            label1="Input / Original", label2="Output / Prediction",
            show_labels=True, make_responsive=True, starting_position=50, in_memory=True
        )
    except Exception:
        left = _pil_to_data_url(orig_img)
        right = _pil_to_data_url(out_img)
        from streamlit.components.v1 import html
        html(f"""
        <link rel="stylesheet" href="https://unpkg.com/img-comparison-slider@7/dist/styles.css">
        <script type="module" src="https://unpkg.com/img-comparison-slider@7/dist/index.js"></script>
        <img-comparison-slider style="width:100%;height:auto;">
          <img slot="first"  style="display:block;max-width:100%;" src="{left}" />
          <img slot="second" style="display:block;max-width:100%;" src="{right}" />
        </img-comparison-slider>
        """, height=520)

# ---------------------------
# CONSENT
# ---------------------------
def ui_consent():
    st.header("Consent")
    agree = st.checkbox("I understand this is a planning tool and not medical advice.", key="consent_agree")
    if st.button("Record Consent", key="consent_button"):
        if not agree:
            st.error("Please agree to continue.")
        else:
            try:
                r = requests.post(f"{API}/consents/", json={"version": "v1"}, headers=headers, timeout=15)
                if r.status_code in (200, 201):
                    st.success("Consent recorded.")
                else:
                    st.error(f"Consent failed: {r.text}")
            except Exception as e:
                st.error(f"Consent error: {e}")

# ---------------------------
# UPLOAD
# ---------------------------
def ui_upload():
    st.header("Upload Photo")
    up = st.file_uploader("Front-facing photo (PNG/JPG)", type=["png", "jpg", "jpeg"], key="upload_input")
    if up and st.button("Save Photo", key="upload_save"):
        files = {"original": (up.name, up.getbuffer(), up.type)}
        try:
            r = requests.post(f"{API}/photos/", files=files, headers=headers, timeout=30)
            if r.status_code == 201:
                st.session_state.work["photo_id"] = r.json()["id"]
                st.success(f"Saved. photo_id={st.session_state.work['photo_id']}")
            else:
                st.error(r.text)
        except Exception as e:
            st.error(f"Upload error: {e}")

# ---------------------------
# PREDICT + COMPARE + DOWNLOAD
# ---------------------------
def ui_predict():
    st.header("Predict")
    pid = st.session_state.work.get("photo_id")
    if not pid:
        st.info("Upload and save a photo first.")
        return

    if st.button("Run Prediction", key="predict_run"):
        try:
            r = requests.post(f"{API}/predictions/", json={"photo_id": pid}, headers=headers, timeout=60)
            if r.status_code == 201:
                st.session_state.work["pred"] = r.json()
                st.success("Prediction complete.")
            else:
                st.error(r.text)
        except Exception as e:
            st.error(f"Prediction error: {e}")

    pred = st.session_state.work.get("pred")
    if not pred:
        return

    # show metrics
    st.subheader("Results")
    st.json(pred.get("metrics", {}))

    # build URLs
    out_url = pred["output"]
    if out_url.startswith("/"):
        out_url = f"http://localhost:8000{out_url}"

    # get original URL via photo detail
    try:
        r = requests.get(f"{API}/photos/{pred['photo']['id']}/", headers=headers, timeout=15)
        r.raise_for_status()
        orig_url = r.json()["original"]
        if orig_url.startswith("/"):
            orig_url = f"http://localhost:8000{orig_url}"
    except Exception as e:
        st.error(f"Could not fetch original photo: {e}")
        return

    # fetch both images
    try:
        orig_img = Image.open(io.BytesIO(requests.get(orig_url, timeout=30).content)).convert("RGB")
        out_img  = Image.open(io.BytesIO(requests.get(out_url,  timeout=30).content)).convert("RGB").resize(orig_img.size)
    except Exception as e:
        st.error(f"Could not load images for comparison: {e}")
        return

    # comparison tabs
    tab1, tab2 = st.tabs(["🧲 Before/After (Slider)", "🎚️ Blend (Opacity)"])
    with tab1:
        image_compare(orig_img, out_img)
    with tab2:
        alpha = st.slider("Blend %", 0, 100, 50, 1, key="blend_slider_main")
        blended = Image.blend(orig_img.convert("RGBA"), out_img.convert("RGBA"), alpha/100.0)
        st.image(blended, caption=f"Blended view ({alpha}%)", use_container_width=True)

    # download predicted only (with username suffix)
    try:
        pred_data = requests.get(out_url, timeout=30).content
        st.download_button(
            "Download Predicted PNG",
            data=pred_data,
            file_name=f"predicted_{_uname_safe}.png",
            mime="image/png",
            key="download_pred_only",
        )
    except Exception as e:
        st.error(f"Download failed: {e}")

# ---------- call the sections ----------
ui_consent()
st.divider()
ui_upload()
st.divider()
ui_predict()