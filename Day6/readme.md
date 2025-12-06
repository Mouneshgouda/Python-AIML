https://holobooth.flutter.dev/#/

https://storage.googleapis.com/tfjs-examples/webcam-transfer-learning/dist/index.html


## h
https://medium.com/@sumbatilinda/deep-learning-part-1-understanding-basic-neural-networks-c9ccdb17a343




## App.py

```python
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Load saved model
model = tf.keras.models.load_model("cat_dog_model.h5")
IMG_SIZE = 64  # match training size

st.title("Cat vs Dog Classifier")
st.write("Upload an image of a cat or dog for prediction.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img_resized = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = float(model.predict(img_array)[0][0])
    label = "Dog" if prediction > 0.5 else "Cat"
    confidence = prediction if prediction > 0.5 else (1 - prediction)

    # Output
    st.write(f"### Prediction: **{label}**")
    st.write(f"### Confidence: **{confidence:.2%}**")

```
