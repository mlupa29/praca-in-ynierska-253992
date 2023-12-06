import tensorflow as tf
import streamlit as st
import numpy as np
from PIL import Image
from io import BytesIO
import boto3
import os


model = tf.keras.models.load_model('FINAL_model_ENETB3_11.h5')

def predict(image):
   
    image = np.array(image.resize((224, 224))) 
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)
    y_pred = np.argmax(prediction, axis=1)
    if y_pred == 0:
        pred = 'ALL'
    else:
        pred = 'HEM'

    return pred

# Funkcja do zapisu obrazu wraz z wynikiem predykcji w AWS S3
def save_image_with_prediction_to_s3(image, prediction, original_filename):
    # Połączenie z S3
    aws_access_key_id = ''
    aws_secret_access_key = ''
    bucket_name = 'praca-inzynierska-mlupa29'

    filename, file_extension = os.path.splitext(original_filename)

    new_filename = f'{filename}_{prediction}{file_extension}'

    s3 = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)

    image_bytes = BytesIO()
    image.save(image_bytes, format='JPEG')
    image_bytes = image_bytes.getvalue()

    s3.put_object(Bucket=bucket_name, Key=new_filename, Body=image_bytes)


st.sidebar.markdown("<h3 style='color: #FFFFFF;'>Opis aplikacji:</h3>", unsafe_allow_html=True)
st.sidebar.markdown("Aplikacja służy do rozpoznawania ostrej białaczki limfoblastycznej na podstawie przesłanych obrazów. Wybierz obraz, wykonaj predykcję lub zapisz predykcję wraz z obrazem w AWS S3.")
st.sidebar.markdown("---")


st.markdown("<h1 style='text-align: center; color: white;'>Aplikacja webowa do rozpoznawania komórek nowotworowych</h1>", unsafe_allow_html=True)


uploaded_image = st.file_uploader("Wybierz obraz...", type=["jpg", "png", "jpeg", "bmp"])


if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Wybrany obraz", use_column_width=True, output_format='JPEG', channels='RGB', width=None, clamp=False)

    # Przycisk do dokonania predykcji
    if st.button("Dokonaj predykcji"):
        prediction = predict(image)
        if prediction == "ALL":
            st.markdown(f"<h2 style='color: white;'>Wynik predykcji: <span style='color: red;'>{prediction}</span></h2>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h2 style='color: white;'>Wynik predykcji: <span style='color: green;'>{prediction}</span></h2>", unsafe_allow_html=True)

    # Przycisk do zapisu obrazu wraz z wynikiem predykcji w AWS S3
    if st.button("Zapisz obraz wraz z predykcją w AWS S3"):
        prediction = predict(image)
        original_filename = uploaded_image.name  
        original_filename = original_filename[:-4]
        save_image_with_prediction_to_s3(image, prediction, original_filename)
        st.write(f"Obraz wraz z wynikiem predykcji ({prediction}) został zapisany w AWS S3 jako '{original_filename}_{prediction}.jpg'.")
