import streamlit as st
import tensorflow as tf
import numpy as np
from skimage.transform import resize
import cv2
from collections import deque
import os
import shutil
import tempfile
import datetime
import atexit
from PIL import Image # For opening image data from st.camera_input

# --- Configuration ---
MODEL_PATH = "gesture_model.keras"
IMAGE_HEIGHT = 120
IMAGE_WIDTH = 120
SEQUENCE_LENGTH = 30 # Your model expects 30 frames.
SEQUENCE_LENGTH_TO_CAPTURE = 30 # <--- ADDED THIS LINE: Number of distinct images the user will capture

GESTURE_LABELS = [
    "Left Swipe", # Assuming label 0
    "Right Swipe",   # Assuming label 1
    "Stop",  # Assuming label 2
    "Thumbs Down", # Assuming label 3
    "Thumbs Up"         # Assuming label 4
]

# --- Global variable for temporary directory ---
if 'temp_dir_path' not in st.session_state:
    temp_dir = tempfile.TemporaryDirectory()
    st.session_state.temp_dir_path = temp_dir.name
    st.session_state.temp_dir_obj = temp_dir # Store the object to prevent it from being garbage collected too early
    atexit.register(temp_dir.cleanup) # Register cleanup when the script exits

# --- Session state for captured frames ---
if 'captured_frames_list' not in st.session_state:
    st.session_state.captured_frames_list = [] # Stores raw image bytes/arrays

# --- Preprocessing Functions (from your training code) ---
def convert_to_grayscale(data):
    return data.mean(axis=-1, keepdims=True)

def normalize_data(data):
    return data / 127.5 - 1

def preprocess_image_for_model(image_array):
    """
    Applies the full preprocessing pipeline to a single image array.
    Expects an RGB image (NumPy array).
    """
    resized_image = resize(image_array, (IMAGE_HEIGHT, IMAGE_WIDTH), anti_aliasing=True)
    grayscale_image = convert_to_grayscale(resized_image)
    normalized_image = normalize_data(grayscale_image * 255.0) # Multiply by 255 before normalizing as resize outputs [0,1]
    final_image = np.stack([normalized_image[:,:,0], normalized_image[:,:,0], normalized_image[:,:,0]], axis=-1)
    return final_image

@st.cache_resource
def load_gesture_model():
    """Loads the pre-trained Keras model and caches it."""
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        st.warning(f"Please ensure '{MODEL_PATH}' is in the same directory as this script.")
        return None

# --- Function to clear captured images and temp folder ---
def clear_captured_images():
    st.session_state.captured_frames_list = []
    output_folder = os.path.join(st.session_state.temp_dir_path, "captured_frames_for_prediction")
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)
    st.success("Captured images cleared. Ready for new capture.")
    st.rerun() # Rerun to clear the displayed images

# --- Streamlit App Layout ---
st.set_page_config(
    page_title="Gesture Recognition App",
    page_icon="👋",
    layout="centered"
)

st.title("👋 Gesture Recognition (Image Capture)")

tab1, tab2 = st.tabs(["🚀 Gesture Predictor", "💡 Methodology Explained"])

with tab1:
    st.header("Gesture Predictor from Captured Images")
    st.write(f"Capture **{SEQUENCE_LENGTH_TO_CAPTURE} sequential images** by clicking 'Take Photo' for prediction.") # Changed wording
    st.markdown("- **Thumbs Up**")
    st.markdown("- **Thumbs Down**")
    st.markdown("- **Left Swipe**")
    st.markdown("- **Right Swipe**")
    st.markdown("- **Stop**")
    st.write(f"The model expects a sequence of **{SEQUENCE_LENGTH} frames**.")

    model = load_gesture_model()

    if model:
        # Camera input widget
        captured_image = st.camera_input("Take Photo", key="camera_input")

        if captured_image is not None:
            # Convert bytes to PIL Image, then to NumPy array (RGB)
            pil_image = Image.open(captured_image)
            rgb_image = np.array(pil_image)

            # Store the raw RGB NumPy array in session state
            st.session_state.captured_frames_list.append(rgb_image)
            st.info(f"Captured image {len(st.session_state.captured_frames_list)}/{SEQUENCE_LENGTH_TO_CAPTURE}")

        st.markdown("---")

        # Display captured images and prepare for prediction
        if st.session_state.captured_frames_list:
            st.subheader(f"Captured Images ({len(st.session_state.captured_frames_list)}/{SEQUENCE_LENGTH_TO_CAPTURE}):")
            
            output_folder_current_run = os.path.join(st.session_state.temp_dir_path, "captured_frames_for_prediction")
            os.makedirs(output_folder_current_run, exist_ok=True) # Ensure folder exists for this run

            cols = st.columns(len(st.session_state.captured_frames_list))
            preprocessed_frames_for_model = [] # Store preprocessed frames for stacking

            for i, img_array in enumerate(st.session_state.captured_frames_list):
                # Display original captured image (or a smaller version)
                with cols[i]:
                    st.image(img_array, caption=f"Original {i+1}", width=100)

                # Preprocess for the model
                processed_frame = preprocess_image_for_model(img_array)
                preprocessed_frames_for_model.append(processed_frame)

                # Save preprocessed image to temporary folder
                # Convert normalized data back to 0-255 for saving
                display_img_for_save = ((processed_frame + 1) * 127.5).astype(np.uint8)
                if display_img_for_save.shape[-1] == 1:
                    display_img_for_save = np.repeat(display_img_for_save, 3, axis=-1)
                
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                frame_filename = os.path.join(output_folder_current_run, f"captured_pred_frame_{i+1}_{timestamp}.png")
                cv2.imwrite(frame_filename, cv2.cvtColor(display_img_for_save, cv2.COLOR_RGB2BGR)) # Save as BGR

            st.markdown(f"**Preprocessed images saved to:** `{output_folder_current_run}`")

            # Prediction logic
            if len(st.session_state.captured_frames_list) == SEQUENCE_LENGTH_TO_CAPTURE:
                st.markdown("---")
                st.subheader("Ready for Prediction:")
                
                # We need to create a sequence of SEQUENCE_LENGTH frames for the model.
                # If SEQUENCE_LENGTH_TO_CAPTURE is less than SEQUENCE_LENGTH,
                # we'll duplicate the last captured frame until we reach SEQUENCE_LENGTH.
                # This is a simplification; for better results, actual video input is recommended.
                final_input_sequence = list(preprocessed_frames_for_model)
                while len(final_input_sequence) < SEQUENCE_LENGTH:
                    if final_input_sequence: # Ensure there's at least one frame to duplicate
                        final_input_sequence.append(final_input_sequence[-1]) # Duplicate the last frame
                    else:
                        break # Should not happen if SEQUENCE_LENGTH_TO_CAPTURE > 0
                
                # Trim if too many (shouldn't happen with current logic but for robustness)
                final_input_sequence = final_input_sequence[:SEQUENCE_LENGTH]

                if st.button("Predict Gesture"):
                    st.write("Analyzing gesture...")
                    try:
                        input_sequence_np = np.stack(final_input_sequence, axis=0)
                        input_sequence_np = np.expand_dims(input_sequence_np, axis=0) # Add batch dimension

                        prediction = model.predict(input_sequence_np, verbose=0)
                        predicted_class_idx = np.argmax(prediction)
                        confidence = np.max(prediction) * 100

                        predicted_gesture = GESTURE_LABELS[predicted_class_idx]

                        st.success(f"**Predicted Gesture: {predicted_gesture}**")
                        st.info(f"Confidence: {confidence:.2f}%")

                    except Exception as e:
                        st.error(f"An error occurred during prediction: {e}")
                        st.write("Please ensure frames were captured correctly.")
                
                st.button("Clear Captured Images", on_click=clear_captured_images) # Button to reset
            elif len(st.session_state.captured_frames_list) > SEQUENCE_LENGTH_TO_CAPTURE:
                 st.warning(f"You have captured more than {SEQUENCE_LENGTH_TO_CAPTURE} images. Please click 'Clear Captured Images' to start over.")
                 st.button("Clear Captured Images", on_click=clear_captured_images)
            else:
                 st.info(f"Please capture {SEQUENCE_LENGTH_TO_CAPTURE - len(st.session_state.captured_frames_list)} more images.")
                 st.button("Clear Captured Images", on_click=clear_captured_images)
        else:
            st.info("No images captured yet. Click 'Take Photo' to begin.")

    else:
        st.error("Model could not be loaded. Please check the console for errors and ensure 'gesture_model.keras' is in the same directory.")


with tab2:
    st.header("💡 Methodology Explained")
    st.write(
        """
        This section delves into the technical details of the gesture recognition system.
        """
    )

    st.subheader("Problem Statement and Gestures")
    st.write(
        """
        The objective is to develop a smart TV feature that recognizes five distinct gestures to control TV functionalities without a remote. Each gesture corresponds to a specific action:

        * **Thumbs Up:** Increase the volume.
        * **Thumbs Down:** Decrease the volume.
        * **Left Swipe:** 'Jump' backwards 10 seconds.
        * **Right Swipe:** 'Jump' forward 10 seconds.
        * **Stop:** Pause the movie.

        Each video in the dataset is a sequence of 30 frames (images).
        """
    )

    st.subheader("Training Dataset (General Characteristics)")
    st.write(
        """
        The model was trained on a dataset consisting of short video sequences, specifically 30 frames per video, representing each of the five gestures. Common characteristics of such datasets for gesture recognition include:

        * **Varied Lighting and Backgrounds:** To ensure robustness, videos are typically captured under different lighting conditions and with diverse backgrounds.
        * **Subject Diversity:** Multiple individuals perform each gesture to account for variations in hand shapes and movement styles.
        * **Image Resolution and Format:** The raw images might come in various resolutions and are often downsampled to a consistent size (e.g., 120x120 pixels in this case) to reduce computational complexity and standardize input for the neural network. They are typically RGB color images.
        * **Data Augmentation:** During training, techniques like rotation, shifting, shearing, zooming, and horizontal flipping are commonly applied to artificially expand the dataset. This helps the model generalize better to unseen data and become more invariant to minor variations in gesture execution.

        Your training code indicates the use of `train.csv` and `val.csv` for managing training and validation data paths, suggesting a structured dataset where each row points to a video folder and its corresponding label.
        """
    )

    st.subheader("Data Preprocessing Pipeline")
    st.write(
        """
        Effective preprocessing is critical to prepare raw video frames for consumption by a deep learning model. Based on your provided training code, the following steps were applied to each frame:

        1.  **Resizing:** All images are resized to a uniform dimension of **120x120 pixels**. This is essential because neural networks, especially convolutional layers, require fixed-size inputs.
        2.  **Grayscale Conversion:** The images are converted from RGB to grayscale. Although the model input tensor maintains 3 channels, the values across these channels are made identical, effectively processing grayscale information. This reduces the dimensionality of the input and can sometimes improve performance by focusing on structural information rather than color.
        3.  **Normalization:** Pixel values, originally ranging from 0 to 255, are normalized to a range of **-1 to 1** using the formula $(data / 127.5) - 1$. This helps in faster and more stable training of neural networks by ensuring input features are on a similar scale.

        For inference in this Streamlit app, frames are captured from the live video stream, and each frame undergoes these same preprocessing steps. A sequence of 30 processed frames is then used as input for the 3D Conv model.
        """
    )

    st.subheader("Model Architecture (3D Convolutional Neural Network)")
    st.write(
        """
        The problem statement specifies building a **3D Convolutional (Conv3D) model**. Unlike traditional 2D CNNs that process single images, 3D CNNs are designed to extract features from volumetric data, such as video sequences.

        A typical 3D CNN architecture for gesture recognition often includes:

        * **3D Convolutional Layers:** These layers apply filters in three dimensions (height, width, and time/depth), allowing the model to learn spatial features within each frame *and* temporal features across consecutive frames. This is crucial for capturing motion patterns that define gestures.
        * **3D Pooling Layers:** These layers reduce the spatial and temporal dimensions of the feature maps, helping to make the model more robust to minor variations and reducing computational load.
        * **Activation Functions:** Rectified Linear Unit (ReLU) is commonly used after convolutional layers to introduce non-linearity.
        * **Flatten Layer:** After several convolutional and pooling layers, the 3D feature maps are flattened into a 1D vector.
        * **Dense (Fully Connected) Layers:** These layers perform classification based on the extracted features.
        * **Output Layer:** A `softmax` activation function is typically used in the final layer to output probabilities for each of the five gesture classes.

        The problem statement highlights the importance of **total number of parameters** for faster inference. This suggests a focus on building an efficient model, potentially by using fewer layers, smaller filters, or techniques like grouped convolutions, to minimize prediction time on the target device (smart TV).
        """
    )

    st.subheader("Training Process")
    st.write(
        """
        The model training typically involves:

        * **Loss Function:** For multi-class classification, `categorical_crossentropy` is a standard choice when labels are one-hot encoded (as indicated by `batch_labels[folder, int(t[...])] = 1`).
        * **Optimizer:** An optimizer (e.g., Adam, SGD, RMSprop) is used to adjust the model's weights during training to minimize the loss function.
        * **Epochs:** The number of times the entire training dataset is passed forward and backward through the neural network. Your training code specified 10 epochs.
        * **Batch Size:** The number of samples (video sequences) processed before the model's internal parameters are updated. Your training code set the batch size to 10.
        * **Validation:** A separate validation dataset is used to monitor the model's performance on unseen data during training, helping to detect overfitting.
        * **Callbacks:** Techniques like `ModelCheckpoint` can be used to save the best performing model based on validation metrics, and `ReduceLROnPlateau` can dynamically adjust the learning rate.
        """
    )

    st.subheader("Inference with Streamlit and Image Capture") # Updated title
    st.write(
        """
        The Streamlit application now uses `st.camera_input` to capture individual images for gesture recognition.

        1.  **Image Capture:** The `st.camera_input` widget allows you to take photos from your webcam.
        2.  **Frame Buffer:** Each captured image is stored in Streamlit's `st.session_state`.
        3.  **Preprocessing:** Each captured image is preprocessed (resized, converted to grayscale-like, normalized) to match the model's input requirements.
        4.  **Sequence Creation:** Once the required number of images (e.g., 5 images) are captured, they are used to construct the input sequence for the 3D Conv model. If fewer images are captured than the model expects (e.g., 5 vs 30), the last captured image is duplicated to fill the sequence.
        5.  **Prediction Trigger:** When the "Predict Gesture" button is clicked, the assembled sequence is fed into the loaded Keras model.
        6.  **Output:** The model's prediction (the most likely gesture and its confidence) is displayed to the user.
        7.  **Temporary Folder Management:** Captured and preprocessed images are saved to a temporary folder (`captured_frames_for_prediction`) which is automatically cleaned up when the Streamlit session ends or manually via the "Clear Captured Images" button.
        """
    )