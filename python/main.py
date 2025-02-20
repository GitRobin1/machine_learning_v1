import streamlit as st
import os
import numpy as np
import random
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.neighbors import NearestNeighbors

st.set_page_config(page_title="Recommandation d'Images", layout="wide")

model = VGG16(weights='imagenet', include_top=False, pooling='avg')

@st.cache_data
def extract_features(image_path):
    """ Extrait les caractéristiques d'une image """
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array)
    return features.flatten()

@st.cache_data
def load_and_extract_features(dataset_dir):
    """ Charge toutes les images et extrait leurs caractéristiques """
    valid_extensions = ('.bmp', '.jpg', '.png')
    image_paths = [os.path.join(dataset_dir, img) for img in os.listdir(dataset_dir) if img.lower().endswith(valid_extensions)]
    feature_list = np.array([extract_features(img) for img in image_paths])
    return image_paths, feature_list

dataset_dir = "dataset/"
image_paths, feature_list = load_and_extract_features(dataset_dir)

knn = NearestNeighbors(n_neighbors=5, metric='euclidean')
knn.fit(feature_list)

if "liked_images" not in st.session_state:
    st.session_state.liked_images = []
if "remaining_images" not in st.session_state:
    st.session_state.remaining_images = image_paths.copy()

st.sidebar.title("Navigation")
page = st.sidebar.radio("Aller à", ["Calibration", "Recommandations"])
if page == "Calibration":
    if "current_image" not in st.session_state:
        st.session_state.current_image = random.choice(st.session_state.remaining_images)

    st.title("Calibration - Sélectionnez vos images préférées")

    if not st.session_state.remaining_images:
        st.warning("Toutes les images ont été vues !")
    else:
        col1, col2, col3 = st.columns([1, 3, 1])  
        with col2:
            st.image(st.session_state.current_image, width=300)

        col4, col5 = st.columns([1, 1])
        with col4:
            if st.button("Like ❤️"):
                st.session_state.liked_images.append(st.session_state.current_image)
                st.session_state.remaining_images.remove(st.session_state.current_image)
                if st.session_state.remaining_images:
                    st.session_state.current_image = random.choice(st.session_state.remaining_images)
                else:
                    st.session_state.current_image = None
                st.rerun()

        with col5:
            if st.button("Skip ⏭️"):
                st.session_state.remaining_images.remove(st.session_state.current_image)
                if st.session_state.remaining_images:
                    st.session_state.current_image = random.choice(st.session_state.remaining_images)
                else:
                    st.session_state.current_image = None
                st.rerun()

    st.write("Images aimées:", st.session_state.liked_images)


elif page == "Recommandations":
    st.title("Images recommandées")
    
    if not st.session_state.liked_images:
        st.warning("Veuillez d'abord calibrer en aimant des images !")
    else:
        liked_features = np.array([extract_features(img) for img in st.session_state.liked_images])
        distances, indices = knn.kneighbors(liked_features, n_neighbors=5)
        recommended_images = {image_paths[i] for idx in indices for i in idx}

        cols = st.columns(5)
        for i, img_path in enumerate(recommended_images):
            with cols[i % 5]:
                st.image(img_path, use_container_width=True)
