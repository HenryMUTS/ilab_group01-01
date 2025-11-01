# 🖼️ Nose Vision AI (FastAPI + Streamlit)

This project combines a **FastAPI backend** and a **Streamlit frontend (UI)** to process user images, run model predictions, and manage submissions via an admin panel.

---

## ⚙️ 1. CodeFormer Setup (Install First)

CodeFormer is used for **face restoration and enhancement**.  
It must be **installed inside the `model/` folder** before running the app.

---

### 🧩 Option A — Conda Installation (Recommended)

```bash
# Go to your model directory
cd models

# Clone the CodeFormer repository
git clone https://github.com/sczhou/CodeFormer
cd CodeFormer

# Create and activate a new Conda environment
conda create -n codeformer python=3.8 -y
conda activate codeformer

# Install dependencies
pip install -r requirements.txt
python basicsr/setup.py develop

# (Optional) For face detection/cropping with dlib
# conda install -c conda-forge dlib
```

---

### 🪄 Option B — Pip-Only Installation (No Conda, No dlib)

If you prefer not to use Conda:

```bash
# Go to your model directory
cd models

# Clone the CodeFormer repository
git clone https://github.com/sczhou/CodeFormer
cd CodeFormer

# Install dependencies with pip
pip install -r requirements.txt
python basicsr/setup.py develop
```

---

### 📦 Download Pretrained Models

#### Option 1 — Manual Download
Download pretrained weights and place them in the correct folders under `CodeFormer/weights/`:

- [Facelib & Dlib (optional)](https://github.com/sczhou/CodeFormer/releases/tag/v0.1.0)
- [CodeFormer Weights](https://github.com/sczhou/CodeFormer/releases/tag/v0.1.0)

#### Option 2 — Script Download

```bash
python scripts/download_pretrained_models.py facelib
python scripts/download_pretrained_models.py CodeFormer
# Skip dlib if not using face detection
```

---

### 🧪 Quick Test (Optional)

Verify that CodeFormer runs properly:

```bash
python inference_codeformer.py -w 0.7 -s 2 --input_path ./inputs --output_path ./results
```

---

## 🚀 2. Setup Environment for the Main App

The main app uses **FastAPI** for the backend and **Streamlit** for the UI.  
You can use either **Poetry** or **pip** to set up dependencies.

---

### 🧩 Option A — Poetry Setup (Recommended)

#### 1️⃣ Install Poetry

```bash
pip install poetry
```

#### 2️⃣ Install Dependencies

```bash
cd ../..
poetry install
```

This will create a virtual environment and install everything automatically  
(FastAPI, Uvicorn, Streamlit, Pillow, NumPy, etc.).

---

### 🪄 Option B — Pip-Only Setup (Alternative)

If you’re not using Poetry, install dependencies manually:

```bash
cd ../..
pip install -r requirments.txt
```

(You may include any additional libraries your project requires.)

---

## 🖥️ 3. Run the FastAPI Backend

#### Using Poetry

```bash
poetry run uvicorn ilab_group01_01.API.model_api:app --reload --host 127.0.0.1 --port 8000
```

#### Using Pip

If you installed packages manually:

```bash
uvicorn ilab_group01_01.API.model_api:app --reload --host 127.0.0.1 --port 8000
```

- The FastAPI app lives in `model_api.py`.
- Prediction endpoint: 👉 [http://127.0.0.1:8000/predict](http://127.0.0.1:8000/predict)

---

## 🎨 4. Run the Streamlit UI

#### Open a new terminal
```bash
start
```
Press Command-N for Mac
#### Using Poetry

```bash
poetry run streamlit run ilab_group01_01/UI/app.py
```

#### Using Pip

If you installed manually:

```bash
streamlit run ilab_group01_01/UI/app.py
```

### UI Features

- 🧍 User form and image upload  
- 🧠 Prediction call to FastAPI backend  
- 🔄 Before/after comparison slider  
- 🧾 Admin page (`pages/admin.py`) for reviewing submissions  

Streamlit will open in your browser automatically:  
👉 [http://localhost:8501](http://localhost:8501)

---

## ⚡ 5. Run Both Components

You must have both services running simultaneously:

- **FastAPI (Backend)** → [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **Streamlit (Frontend)** → [http://localhost:8501](http://localhost:8501)

---

## 📂 Recommended Folder Structure

```bash
project_root/
│
├── ilab_group01_01/
│   ├── API/
│   │   └── model_api.py
│   ├── UI/
│   │   ├── app.py
│   │   └── pages/
│   │       └── admin.py
│
├── model/
│   └── CodeFormer/
│       ├── basicsr/
│       ├── weights/
│       │   ├── facelib/
│       │   └── CodeFormer/
│       └── scripts/
│
└── pyproject.toml  or  requirements.txt
```

