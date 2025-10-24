# Streamlit UI

import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from keras.layers import Input, TFSMLayer
from keras.models import Model

st.title("Tuberculosis X-ray Classification")

st.write("""
**Tuberculosis (TB)** is an infectious disease caused by the *Mycobacterium tuberculosis* bacterium.
It is a major global health challenge, with millions affected every year. Early diagnosis and treatment are crucial in controlling its spread.
""")

st.write("""
### The Role of Chest X-rays:
Chest X-rays are a vital diagnostic tool in identifying TB. In this app, we use deep learning models to analyze X-ray images and classify them as either 'TB-positive' or 'TB-negative'. This process helps doctors make faster and more accurate diagnoses, enabling timely treatment.
""")

normal_text = """
            Great news — your chest X-ray shows no signs of tuberculosis (TB).  
            There are no visible abnormalities or signs of infection associated with TB at the moment.
            
            However, if you experience symptoms like a persistent cough, fever, or chest pain, 
            consider getting a checkup with your healthcare provider.
            """

tb_text = """
            We have detected signs consistent with tuberculosis (TB) in your chest X-ray.  
            While this result doesn't confirm TB with 100% certainty (as further medical tests are required),
            it's crucial to seek medical attention as soon as possible.

            Early detection and treatment are key to recovery and preventing the spread of the disease.
            Please contact a healthcare professional immediately for further diagnostic tests.
            """

# Model
model_path = "models/finetuned_efficientnet"
input_layer = Input(shape=(128,128,3))
tfsm_layer = TFSMLayer(model_path, call_endpoint='serving_default')
output_layer = tfsm_layer(input_layer)

model = Model(inputs=input_layer, outputs=output_layer)

# File uploader
file = st.file_uploader("Upload X-Ray Image", type=["jpg", "jpeg", "png"])

if file:
    # Read image using OpenCV
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)  # Read as BGR
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)        # Convert to RGB

    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    img_resized = cv2.resize(img, (128, 128))  # Resize
    img_resized = img_resized.astype(np.float32)
    img_resized = tf.keras.applications.efficientnet.preprocess_input(img_resized)
    img_resized = np.expand_dims(img_resized, axis=0)  # Add batch dimension

    # Prediction
    predicted = model.predict(img_resized)

    # Prediction button
    if st.button("Click to see the prediction"):
        with st.spinner("Analyzing..."):
            predicted = model.predict(img_resized)
            pred = list(predicted.values())[0]  # Since ipredicted is a dictionary

            # Now pred is a NumPy array
            if pred[0][0] < 0.5:
                st.subheader("✅ Your Chest X-ray is Normal!")
                st.write(normal_text)
                st.info("Stay healthy and take care of your lungs!")
            else:
                st.subheader("⚠️ TB Detected")
                st.write(tb_text)
                st.error("Don't wait — seek help as soon as possible. Your health is the top priority!")
else:
    st.warning("Kindly upload an X-Ray image.")
