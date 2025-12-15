import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Alzheimer's Detection",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Alzheimer's Disease Detection")
st.write("Upload a brain MRI image to predict Alzheimer's stage")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_alzheimer_model():
    return load_model("Alzheimer.h5")

model = load_alzheimer_model()

# ---------------- CLASS LABELS ----------------
class_names = [
    "Mild Demented",
    "Moderate Demented",
    "Non Demented",
    "Very Mild Demented"
]

# ---------------- IMAGE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "Upload Brain MRI Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img = img.resize((224, 224))  # match training size
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    if st.button("🔍 Predict"):
        preds = model.predict(img_array)
        idx = np.argmax(preds)
        confidence = preds[0][idx] * 100

        st.success(f"🧠 Prediction: **{class_names[idx]}**")
        st.info(f"Confidence: **{confidence:.2f}%**")

        st.subheader("📊 Class Probabilities")
        for i, label in enumerate(class_names):
            st.write(f"{label}: {preds[0][i]*100:.2f}%")
