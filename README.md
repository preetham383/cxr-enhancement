# CXR Enhancement & Pneumonia Detection

This project focuses on enhancing Chest X-ray (CXR) images and training deep learning models for pneumonia detection.

## 🔍 Features
- Image enhancement techniques for X-ray preprocessing
- CNN-based training pipelines
- Model evaluation and ROC analysis
- Web apps using Gradio and Streamlit

## 🗂 Project Structure
.
├── train_cnn.py
├── train_cnn_enhanced.py
├── prepare_dataset.py
├── split_dataset.py
├── evaluate_models.py
├── evaluate_roc.py
├── gradio_app.py
├── streamlit_app.py
├── requirements.txt
└── .gitignore


## 📊 Dataset
Datasets are **not included** in this repository.

You can use publicly available datasets such as:
- Kaggle Chest X-Ray Pneumonia Dataset

Place datasets locally and update paths in the scripts.

## ⚙️ Installation
pip install -r requirements.txt
▶️ Usage
Train a model:

python train_cnn.py
Run Gradio app:

python gradio_app.py
Run Streamlit app:

streamlit run streamlit_app.py
📌 Note
Trained models and images are intentionally excluded to keep the repository lightweight and professional.


## STEP 3: Commit
git add README.md
git commit -m "Add project README"
git push
