import streamlit as st
import face_recognition
import numpy as np
from PIL import Image
import joblib
import cv2

def recognize_faces(image, known_encodings, known_labels, tolerance=0.6):
    image_np = np.array(image.convert("RGB"))
    face_locations = face_recognition.face_locations(image_np)
    face_encodings = face_recognition.face_encodings(image_np, face_locations)

    results = []
    annotated_image = image_np.copy()

    for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
        distances = face_recognition.face_distance(known_encodings, encoding)
        best_index = np.argmin(distances)
        best_distance = distances[best_index]
        label = known_labels[best_index] if best_distance <= tolerance else "Unknown"

        cv2.rectangle(annotated_image, (left, top), (right, bottom), (0, 255, 0), 3)
        cv2.putText(annotated_image, label, (left, top - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        results.append((label, best_distance))

    return Image.fromarray(annotated_image), results


def main():
    st.set_page_config(
        page_title="42 Amman",
        page_icon="🧑‍🦰",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    with st.sidebar:
        st.header("Info")
        st.markdown(
            """
            - Upload JPG, JPEG, PNG images  
            - App uses pretrained face encodings for 42 Amman students  
            - © 2025 Galhajaj
            """
        )

    st.title("🧑‍🦰 Face Recognition for 42 Amman Students")

    uploaded_file = st.file_uploader(
        "Upload an image", type=["jpg", "jpeg", "png"],
        help="Choose an image file to detect and recognize faces"
    )

    if uploaded_file:
        try:
            image = Image.open(uploaded_file)
        except Exception:
            st.error("⚠️ Could not open the image. Please upload a valid JPG, JPEG, or PNG file.")
            return

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)

        @st.cache_resource
        def load_data():
            encodings = joblib.load("encodings.pkl")
            labels = joblib.load("labels.pkl")
            return encodings, labels

        known_encodings, known_labels = load_data()

        with st.spinner("Detecting faces... please wait"):
            annotated_image, results = recognize_faces(image, known_encodings, known_labels, tolerance=0.6)

        with col2:
            st.subheader(f"Results — {len(results)} Face(s) Detected")
            if results:
                st.image(annotated_image, use_container_width=True)
                for i, (label, dist) in enumerate(results, start=1):
                    if label == "Unknown":
                        status = "❓ Unknown person — Not a student in 42 Amman 👀"
                        st.markdown(f"**Face {i}:** {status}")
                    else:
                        status = "✅ is a student in 42 Amman 😎"
                        st.markdown(f"**Face {i}:** {label} — {status}")
            else:
                st.warning("No faces detected. Try a different image. 👀")

    else:
        st.info("Upload an image to start face recognition.")

if __name__ == "__main__":
    main()
