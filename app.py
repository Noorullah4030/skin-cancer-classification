import streamlit as st
import numpy as np
from PIL import Image
import os

# Set page configuration
st.set_page_config(
    page_title="Skin Cancer Classification",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished layout
st.markdown("""
<style>
    .reportview-container {
        background: #f8f9fa;
    }
    .main-header {
        font-family: 'Inter', sans-serif;
        color: #1a1a1a;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }
    .sub-header {
        font-family: 'Inter', sans-serif;
        color: #555555;
        text-align: center;
        font-size: 1.1rem;
        margin-bottom: 30px;
    }
    .class-card {
        padding: 15px;
        border-radius: 8px;
        background-color: #ffffff;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 10px;
        border-left: 5px solid #007bff;
    }
    .prediction-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #111;
        margin-top: 15px;
    }
    .probability-label {
        font-weight: 500;
        margin-bottom: 2px;
    }
</style>
""", unsafe_allow_html=True)

# Define classes and styling colors
CLASSES = [
    "Basal Cell Carcinoma",
    "Melanoma",
    "Normal Skin",
    "Benign"
]

COLORS = {
    "Basal Cell Carcinoma": "#E74C3C", # Red accent
    "Melanoma": "#8E44AD",             # Purple accent
    "Normal Skin": "#2ECC71",          # Green accent
    "Benign": "#3498DB"                # Blue accent
}

# Sidebar configuration
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2877/2877239.png", width=80)
st.sidebar.title("App Information")
st.sidebar.markdown("""
### Trained Model Specs
* **Inference Engine:** TensorFlow Lite
* **Model Type:** float32 classification model
* **Input Resolution:** 224x224 (RGB)
* **Underlying Architecture:** CNN
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### Preprocessing Pipeline Steps")
st.sidebar.code("""
1. Resize: (224, 224)
2. Scale: / 255.0 (ToTensor)
3. Normalize:
   mean = [0.485, 0.456, 0.406]
   std  = [0.229, 0.224, 0.225]
4. Dimension Expansion:
   shape to (1, 224, 224, 3)
""")

# Model Path Constants
MODEL_PATH = "assets/skin_cancer_float32.tflite"

@st.cache_resource
def load_tflite_model(model_path):
    """Loads the TFLite Interpreter using robust loading mechanics and fallbacks."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at local path: '{model_path}'. "
                                f"Please make sure you have placed the trained TFLite model in the assets/ directory.")
    
    # Attempt to load using standard tensorflow interpreter first, then fall back to tflite_runtime
    try:
        import tensorflow as tf
        interpreter = tf.lite.Interpreter(model_path=model_path)
    except ImportError:
        try:
            import tflite_runtime.interpreter as tflite
            interpreter = tflite.Interpreter(model_path=model_path)
        except ImportError as e:
            raise ImportError("Could not import either 'tensorflow' or 'tflite-runtime'. "
                              "Please install at least one of these libraries as per instructions.") from e
            
    interpreter.allocate_tensors()
    return interpreter

def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Applies EXACTLY the same preprocessing pipeline used during training.
    Transforms:
      - transforms.Resize((224,224))
      - transforms.ToTensor() -> scales to [0,1]
      - transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    """
    # 1. Ensure RGB mode
    img = image.convert("RGB")
    
    # 2. Resize to (224, 224) using bilinear interpolation
    try:
        resample_mode = Image.Resampling.BILINEAR
    except AttributeError:
        resample_mode = Image.BILINEAR  # Older Pillow version fallback
    img_resized = img.resize((224, 224), resample=resample_mode)
    
    # 3. Convert to np.float32 and scale to [0.0, 1.0] (Equivalent to transforms.ToTensor())
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    
    # 4. Normalize channels using ImageNet mean and standard deviations
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_normalized = (img_array - mean) / std
    
    # 5. Expand dimensions to (1, 224, 224, 3) to introduce the batch dimension
    img_batch = np.expand_dims(img_normalized, axis=0)
    return img_batch

def softmax(logits: np.ndarray) -> np.ndarray:
    """Computes the mathematically precise softmax activation values for logits array."""
    # Stability correction subtraction prevents numeric overflow
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

# Main Application Header
st.markdown("<h1 class='main-header'>🩺 Skin Cancer Classification</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Upload a dermatoscopic lesion photo to compute real-time classification predictions and probabilities using the trained TFLite model.</p>", unsafe_allow_html=True)

# Try loading the model with clear user feedback
interpreter = None
try:
    interpreter = load_tflite_model(MODEL_PATH)
    st.success("✅ Model file 'skin_cancer_float32.tflite' loaded successfully!")
except FileNotFoundError as e:
    st.warning("⚠️ Local trained model is missing.")
    st.info(f"""
    **Getting Started:**
    The application is currently waiting for your model.
    1. Create a folder named `assets/` in your project root.
    2. Upload raw `{MODEL_PATH}` into it.
    3. The application will automatically detect and load the model.
    """)
    st.error(str(e))
except Exception as e:
    st.error(f"❌ Error loading the TensorFlow Lite model: {str(e)}")

# File Uploader
uploaded_file = st.file_uploader("Choose a dermatoscopic skin image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. Handle Invalid Images check
    try:
        image = Image.open(uploaded_file)
        # Attempt simple loading/rotation touch to verify content validity
        image.verify()
        # Re-open after verfication since verify closes the object handle
        image = Image.open(uploaded_file)
        
        # Display Image 
        st.subheader("🖼️ Uploaded Skin Lesion Asset")
        st.image(image, use_container_width=True, caption="Source Dermatoscopy Image")
        
    except Exception as e:
        st.error(f"❌ Invalid Image: The uploaded asset could not be opened or is corrupted. Details: {str(e)}")
        uploaded_file = None

# Run Inference if Image is loaded successfully and model interpreter is active
if uploaded_file is not None and interpreter is not None:
    # Inference Button
    if st.button("🚀 Analyze Skin Lesion", use_container_width=True):
        with st.spinner("⏳ Performing precise digital preprocessing and inferring..."):
            try:
                # 1. Get input/output tensor specs
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()
                
                # Check model dimensions
                expected_shape = list(input_details[0]['shape']) # e.g. [1, 224, 224, 3]
                
                # 2. Preprocess our uploaded image
                preprocessed_data = preprocess_image(image)
                
                # 3. Double check shape match
                if list(preprocessed_data.shape) != expected_shape:
                    st.error(f"Inputs mismatch! Model expects tensor of shape {expected_shape}, but generated preprocessed tensor is of shape {list(preprocessed_data.shape)}")
                    st.stop()
                
                # 4. Set Input Tensor
                interpreter.set_tensor(input_details[0]['index'], preprocessed_data)
                
                # 5. Run Interpreter Inference
                interpreter.invoke()
                
                # 6. Retrieve outputs (logits)
                raw_outputs = interpreter.get_tensor(output_details[0]['index'])
                
                # 7. Apply Softmax to raw logits
                probabilities = softmax(raw_outputs)[0] # Grab first index since batch size is 1
                
                # 8. Calculate Predicted Class details
                class_index = np.argmax(probabilities)
                predicted_class = CLASSES[class_index]
                confidence_pct = probabilities[class_index] * 100
                predicted_color = COLORS.get(predicted_class, "#333333")
                
                # Present Prediction Results
                st.markdown("---")
                st.subheader("📊 Primary Analysis Report")
                
                st.markdown(f"""
                <div style="background-color: #fcfcfc; border-left: 6px solid {predicted_color}; padding: 20px; border-radius: 6px; box-shadow: 0 4px 6px rgba(0,0,0,0.02)">
                    <span style="font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.1em; color: #666;">DIAGNOSTIC prediction</span>
                    <h2 style="margin: 0; color: #111; font-weight: 700;">{predicted_class}</h2>
                    <p style="margin: 5px 0 0 0; font-size: 1.25rem; font-weight: 500; color: {predicted_color}">
                        Confidence: <strong>{confidence_pct:.2f}%</strong>
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # 9. Probabilities as progress bars
                st.markdown("### Probabilities Across All Classes")
                cols = st.columns(2)
                for i, class_name in enumerate(CLASSES):
                    probability = float(probabilities[i])
                    # Pick column
                    col_index = i % 2
                    with cols[col_index]:
                        st.markdown(f"""
                        <div style="margin-top: 10px;">
                            <span class="probability-label">{class_name}</span>
                        </div>
                        """, unsafe_allow_html=True)
                        st.progress(probability)
                        st.markdown(f"**Value:** {probability*100:.2f}%")
                
                # High Risk Warnings and Guidance
                if predicted_class in ["Melanoma", "Basal Cell Carcinoma"]:
                    st.warning("""
                    ⚠️ **Attention Needed (High-Risk Classification):** 
                    The model classified this lesion as potentially malignant. Please seek consulting from a board-certified dermatologist for a clinical evaluation and potential histopathological confirmation.
                    """)
                else:
                    st.info("""
                    ℹ️ **Clinical Note:** 
                    Even though benign/normal skin was predicted, always practice sunscreen protection and regularly inspect your moles using the ABCDE guidelines. Consult professionals for any changing or bleeding lesions.
                    """)
                    
            except Exception as e:
                st.error(f"❌ Error during model execution/inference: {str(e)}")
                st.exception(e)
else:
    if uploaded_file is not None and interpreter is None:
         st.error("❌ Classification suspended because the AI model is not successfully loaded. Please verify the `.tflite` model asset in your sidebar setup.")
