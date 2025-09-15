Quick Start
1) Backend (Django API)

cd rhino-app/backend

# Create & activate a virtualenv
python -m venv .venv
source .venv/bin/activate

# Install deps
pip install -r requirements.txt

# Env (copy template if provided)
cp .env.example .env
# Ensure at minimum:
# DJANGO_SECRET_KEY=dev
# DJANGO_DEBUG=True
# DJANGO_ALLOWED_HOSTS=*
# MEDIA_ROOT=.media
# MEDIA_URL=/media/
# CORS_ALLOWED_ORIGINS=http://localhost:8501,http://127.0.0.1:8501

# Migrate DB and create admin
python manage.py makemigrations core
python manage.py migrate
python manage.py createsuperuser

# Run API
python manage.py runserver 0.0.0.0:8000

2) Frontend (Streamlit UI)

cd rhino-app/frontend

# Create & activate a virtualenv
python -m venv .venv
source .venv/bin/activate

# Install deps
# If you have a frontend/requirements.txt, use that:
# pip install -r requirements.txt
pip install streamlit requests pillow
# Optional: for the slider component
pip install "streamlit-image-comparison==0.0.4"

# Run UI (point to your API)
API_BASE=http://127.0.0.1:8000/api python -m streamlit run streamlit_app.py

## Using the App

1. Sign up a new account (username, email, password, DOB, gender, etc.).

2. Log in → the app shows Consent / Upload / Predict sections.

3. Consent → Upload photo → Predict.

4. View Before/After and Blend tabs; Download the predicted PNG.

5. Logout via the top-right button or sidebar.