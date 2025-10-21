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
Start the API server with:
```bash
poetry run uvicorn ilab_group01_01.API.model_api:app --reload --host 127.0.0.1 --port 8000
```
- api.py contains the FastAPI app.

- The prediction endpoint will be available at:

👉 http://127.0.0.1:8000/predict

## 🎨 3. Run the Streamlit UI

Start Streamlit inside Poetry:
```bash
poetry run streamlit run ilab_group01_01/UI/app.py
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


## CodeFormer Install
### Dependencies and Installation

- Pytorch >= 1.7.1
- CUDA >= 10.1
- Other required packages in `requirements.txt`
```
# git clone this repository
git clone https://github.com/sczhou/CodeFormer
cd CodeFormer

# create new anaconda env
conda create -n codeformer python=3.8 -y
conda activate codeformer

# install python dependencies
pip3 install -r requirements.txt
python basicsr/setup.py develop
conda install -c conda-forge dlib (only for face detection or cropping with dlib)
```
<!-- conda install -c conda-forge dlib -->

### Quick Inference

#### Download Pre-trained Models:
Download the facelib and dlib pretrained models from [[Releases](https://github.com/sczhou/CodeFormer/releases/tag/v0.1.0) | [Google Drive](https://drive.google.com/drive/folders/1b_3qwrzY_kTQh0-SnBoGBgOrJ_PLZSKm?usp=sharing) | [OneDrive](https://entuedu-my.sharepoint.com/:f:/g/personal/s200094_e_ntu_edu_sg/EvDxR7FcAbZMp_MA9ouq7aQB8XTppMb3-T0uGZ_2anI2mg?e=DXsJFo)] to the `weights/facelib` folder. You can manually download the pretrained models OR download by running the following command:
```
python scripts/download_pretrained_models.py facelib
python scripts/download_pretrained_models.py dlib (only for dlib face detector)
```

Download the CodeFormer pretrained models from [[Releases](https://github.com/sczhou/CodeFormer/releases/tag/v0.1.0) | [Google Drive](https://drive.google.com/drive/folders/1CNNByjHDFt0b95q54yMVp6Ifo5iuU6QS?usp=sharing) | [OneDrive](https://entuedu-my.sharepoint.com/:f:/g/personal/s200094_e_ntu_edu_sg/EoKFj4wo8cdIn2-TY2IV6CYBhZ0pIG4kUOeHdPR_A5nlbg?e=AO8UN9)] to the `weights/CodeFormer` folder. You can manually download the pretrained models OR download by running the following command:
```
python scripts/download_pretrained_models.py CodeFormer
```


