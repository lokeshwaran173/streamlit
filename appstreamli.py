import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load your pre-trained model here
model = tf.keras.models.load_model('D:/Desktop Loki/HerbScan/plant identification model/plant_identification_model_leaf.h5')

# Define your label mapping dictionary
label_mapping = {0: 'Aloevera', 1: 'Amla', 2: 'Amruthaballi', 3: 'Arali', 4: 'Astma_weed', 5: 'Badipala', 6: 'Balloon_Vine', 7: 'Bamboo', 8: 'Beans', 9: 'Betel', 10: 'Bhrami', 11: 'Bringaraja', 12: 'Caricature', 13: 'Castor', 14: 'Catharanthus', 15: 'Chakte', 16: 'Chilly', 17: 'Citron lime (herelikai)', 18: 'Coffee', 19: 'Common rue(naagdalli)', 20: 'Coriender', 21: 'Curry', 22: 'Doddpathre', 23: 'Drumstick', 24: 'Ekka', 25: 'Eucalyptus', 26: 'Ganigale', 27: 'Ganike', 28: 'Gasagase', 29: 'Ginger', 30: 'Globe Amarnath', 31: 'Guava', 32: 'Henna', 33: 'Hibiscus', 34: 'Honge', 35: 'Insulin', 36: 'Jackfruit', 37: 'Jasmine', 38: 'Kambajala', 39: 'Kasambruga', 40: 'Kohlrabi', 41: 'Lantana', 42: 'Lemon', 43: 'Lemongrass', 44: 'Malabar_Nut', 45: 'Malabar_Spinach', 46: 'Mango', 47: 'Marigold', 48: 'Mint', 49: 'Neem', 50: 'Nelavembu', 51: 'Nerale', 52: 'Nooni', 53: 'Onion', 54: 'Padri', 55: 'Palak(Spinach)', 56: 'Papaya', 57: 'Parijatha', 58: 'Pea', 59: 'Pepper', 60: 'Pomoegranate', 61: 'Pumpkin', 62: 'Raddish', 63: 'Rose', 64: 'Sampige', 65: 'Sapota', 66: 'Seethaashoka', 67: 'Seethapala', 68: 'Spinach1', 69: 'Tamarind', 70: 'Taro', 71: 'Tecoma', 72: 'Thumbe', 73: 'Tomato', 74: 'Tulsi', 75: 'Turmeric', 76: 'ashoka', 77: 'camphor', 78: 'kamakasturi', 79: 'kepala'}

def preprocess_image(image):
    image = image.resize((224, 224))
    image_array = np.array(image)
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def process_predictions(predictions):
    predicted_label_index = np.argmax(predictions)
    predicted_label = label_mapping.get(predicted_label_index, 'Unknown')
    confidence = predictions[0][predicted_label_index]
    return f'Predicted Label: {predicted_label}, Confidence: {confidence:.2f}'

def main():
    st.title('Plant Classification App')

    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        if st.button('Classify'):
            preprocessed_image = preprocess_image(image)
            predictions = model.predict(preprocessed_image)
            result = process_predictions(predictions)
            st.write(result)

if __name__ == '__main__':
    main()
