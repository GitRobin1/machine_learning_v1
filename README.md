# Image Recommendation System with Streamlit and KNN

This is an interactive image recommendation system built using **Streamlit**, **TensorFlow**, and **K-Nearest Neighbors (KNN)**. The application allows users to calibrate their preferences by liking images, and then recommends similar images based on those preferences.

## Project Overview

The system uses a pre-trained **VGG16** model to extract features from images and then utilizes the **K-Nearest Neighbors (KNN)** algorithm to find similar images. The application works in two main phases:

1. **Calibration Phase**: Users view images and select their favorite ones. This helps the system learn the user's preferences.
2. **Recommendation Phase**: Based on the images the user has liked, the system recommends similar images from the dataset.

## Requirements

To run this project, you need the following Python libraries:

- **Streamlit**: For building the interactive web interface.
- **TensorFlow/Keras**: For using the pre-trained VGG16 model to extract image features.
- **scikit-learn**: For the K-Nearest Neighbors (KNN) algorithm.
- **Pillow**: For image manipulation.
- **NumPy**: For array manipulation.

You can install the required libraries using the following command:

```bash
pip install streamlit tensorflow scikit-learn numpy pillow
```

# How to Run the Application

1. Clone this repository

```bash
git clone https://github.com/GitRobin1/machine_learning_v1.git
```

2. Navigate to the project directory

```bash
cd machine_learning_v1
```

3. Place your image dataset inside the dataset/ folder.

4. Run the Streamlit app:

```bash
streamlit run app.py
```

5. Open your web browser and go to http://localhost:8501 to access the application.

## Application Features

### 1. Calibration Phase
In this phase, you will be shown random images from the dataset. For each image, you can either:

- **Like it (❤️)** if you find it interesting.
- **Skip it (⏭️)** if you're not interested.

Your liked images will be stored, and once you've finished selecting your preferences, you can move to the next phase.

### 2. Recommendation Phase
Once you've calibrated by liking some images, the application will recommend similar images based on your preferences. The system uses the **KNN algorithm** to find the 5 most similar images to the ones you liked, and displays them.

## Algorithm Details

### VGG16 Model
The system uses the pre-trained **VGG16** model from Keras to extract features from images. These features are essentially representations of the content of the image that can be used to compare it with other images.

### K-Nearest Neighbors (KNN)
The **KNN algorithm** is used to find the nearest neighbors (most similar images) based on the features extracted by the **VGG16 model**. The algorithm calculates the Euclidean distance between feature vectors to identify similar images.

## Code Explanation

- **Feature Extraction**: The **VGG16 model** is used to extract features from each image in the dataset. This is done in the `extract_features` function.
  
- **KNN Model**: The **NearestNeighbors algorithm** is trained with the features of all images in the dataset. The KNN model is then used to find the 5 closest images to those liked by the user.

- **Session State**: Streamlit's `st.session_state` is used to store the user's liked images, remaining images, and current state of the application across different pages.
