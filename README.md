# 🖼️ Image Prediction App (FastAPI + Streamlit)

This project combines a **FastAPI backend** and a **Streamlit frontend (UI)** to process user images, run model predictions, and manage submissions via an admin panel.

---

## 🚀 1. Setup Environment

We use **Poetry** for dependency management.

### Install Poetry (if not already installed)
```bash
pip install poetry
```

### Install dependencies
```bash
poetry install
```
This creates a virtual environment and installs all dependencies (FastAPI, Uvicorn, Streamlit, Pillow, NumPy, etc.).

## 🖥️ 2. Run the FastAPI Backend
Go to API Directory
```bash
cd ../API
```
Start the API server with:
```bash
poetry run uvicorn API.model_api:app --reload --host 127.0.0.1 --port 8000
```
- api.py contains the FastAPI app.

- The prediction endpoint will be available at:

👉 http://127.0.0.1:8000/predict

## 🎨 3. Run the Streamlit UI

Start Streamlit inside Poetry:
```bash
poetry run streamlit run UI/app.py
```

The main UI (app.py) provides:
- User info form
- Image upload
- Prediction via API
- Comparison slider / blend view

The admin page (pages/admin.py) provides:
- Review of submitted records
- User info alongside before/after images
- Option to approve and save submissions

Streamlit will open in your browser:
👉 http://localhost:8501

## ⚡ 4. Notes
Run both API and UI at the same time:
- API (FastAPI) → port 8000
- UI (Streamlit) → port 8501

If you see numpy.core.multiarray failed to import, reset NumPy:
```bash
poetry run pip uninstall -y numpy
poetry add numpy@latest
```
Approved submissions are saved under saved_records/:
- Images (*_input.png, *_output.png)
- User info (details.txt)