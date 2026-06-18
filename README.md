# 🩺 Skin Cancer Classification Streamlit Web Application

This repository contains a professional Streamlit web application for real-time, in-browser classification of skin lesion photos using your pre-trained TensorFlow Lite model. 

The application is engineered to preprocess images with pixel-perfect accuracy, ensuring complete parity with the PyTorch model's training pipeline so that predictions are 100% consistent with your original training results.

---

## 🔬 How We Guarantee Processing Match (Mathematical Consistency)

When using deep learning models trained in Python with frameworks like PyTorch (`torchvision.transforms`), it is critical that inference inputs match training inputs down to the floating-point values of every pixel.

Our Streamlit code translates your original training pipeline exactly:

```python
# TRAINING PIPELINE:
transforms.Resize((224,224))
transforms.ToTensor()
transforms.Normalize(
    mean=[0.485,0.456,0.406],
    std=[0.229,0.224,0.225]
)
```

Into **mathematically identical** operations using standard scientific Python tools (`PIL` and `NumPy`):

1. **Resolution & Interpolation**: We open images using `PIL.Image` and resize to exactly `(224, 224)` utilizing **Bilinear Interpolation** (`Image.Resampling.BILINEAR`), which is the default PyTorch interpolation method. This ensures anti-aliasing matches exactly.
2. **`ToTensor()` Scaling**: PyTorch converts standard 8-bit RGB channels $[0, 255]$ into floating-point tensors in the range $[0.0, 1.0]$. We replicate this by casting our NumPy array to `np.float32` and dividing by `255.0`.
3. **Channel-Wise Standardization**: We standardize each channel ($C_{\text{norm}} = \frac{C - \text{mean}}{\text{std}}$) matching the ImageNet statistics:
   * **Mean values**: `[0.485, 0.456, 0.406]`
   * **Standard Desviations**: `[0.229, 0.224, 0.225]`
4. **Reshape to Channel-Last & Mini-Batch**: Your TFLite model expects input shape `(1, 224, 224, 3)`. We apply `np.expand_dims(..., axis=0)` to create the virtual batch dimension expected by TensorFlow Lite.

---

## 💻 Manual Setup & Running Locally

Follow these steps to run the Streamlit app on your local computer:

### Prerequisite
Make sure you have **Python 3.8 to 3.11** installed on your workstation.

### Step 1: Clone or copy files
Ensure your project contains:
* `app.py`
* `requirements.txt`
* `assets/skin_cancer_float32.tflite` (Your pre-trained TFLite model file)

### Step 2: Establish a Virtual Environment
It is highly recommended to run the app in an isolated environment as a best practice:
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS / Linux:
source venv/bin/activate
# On Windows Command Prompt:
venv\Scripts\activate
# On Windows PowerShell:
.\venv\Scripts\Activate.ps1
```

### Step 3: Install Required Libraries
Install the lightweight dependencies listed in our file:
```bash
pip install -r requirements.txt
```

*Note on wheels: `tflite-runtime` is used instead of the complete tensorflow framework. This package weighs just ~4MB instead of ~500MB, letting your app run fast and initialize instantly!*

### Step 4: Run the Streamlit Application
Fire up the local host server:
```bash
streamlit run app.py
```
This command automatically serves the app and opens the dashboard in your default browser at `http://localhost:8501`.

---

## ☁️ Deploying to Streamlit Cloud

Streamlit Community Cloud is a free hosting service for public Streamlit apps. Follow these steps to host your Skin Cancer Classifier on the internet in minutes:

### Step 1: Push Code to GitHub
1. Create a public repository on your [GitHub](https://github.com) account.
2. Push your files, maintaining the folder structure:
   ```text
   ├── app.py
   ├── requirements.txt
   └── assets/
       └── skin_cancer_float32.tflite
   ```

### Step 2: Configure Streamlit Cloud Account
1. Visit [share.streamlit.io](https://share.streamlit.io) and log in using your GitHub account credentials.
2. Click **"New App"** in the top-right corner.

### Step 3: Connect Repo & Deploy
1. **Repository**: Select your skin-cancer repository from the dropdown list.
2. **Branch**: Choose your main deployment branch (usually `main` or `master`).
3. **Main file path**: Type `app.py`.
4. Click **"Deploy"**.

Streamlit Cloud installs any packages declared in `requirements.txt` instantly, builds your container, and deploys a live webpage with a shareable secure URL (`https://<app-name>.streamlit.app`)!
