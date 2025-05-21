import streamlit as st
import face_recognition
import numpy as np
from PIL import Image
import joblib
import cv2
import io

def recognize_faces(image, known_encodings, known_labels, tolerance=0.6):
    image_np = cv2.resize(np.array(image.convert("RGB")), (640, 480))
    # Optional: Grayscale for faster processing (uncomment if needed)
    # image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    # image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
    face_locations = face_recognition.face_locations(image_np, model="hog")
    face_encodings = face_recognition.face_encodings(image_np, face_locations)

    results = []
    annotated_image = image_np.copy()

    for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
        distances = face_recognition.face_distance(known_encodings, encoding)
        best_index = np.argmin(distances)
        best_distance = distances[best_index]
        label = known_labels[best_index] if best_distance <= tolerance else "Unknown"

        cv2.rectangle(annotated_image, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(annotated_image, label, (left, top - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        results.append((label, best_distance))

    return Image.fromarray(annotated_image), results

def main():
    st.set_page_config(
        page_title="42 Amman Face Recognition",
        page_icon="🧑‍🦰",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    with st.sidebar:
        st.header("About")
        st.markdown(
            """
            - Upload JPG, JPEG, or PNG images
            - Recognizes 42 Amman students using pre-trained face encodings
            - Adjust tolerance for recognition sensitivity
            - © 2025 Galhajaj
            """
        )
        tolerance = st.slider("Recognition Tolerance", 0.3, 0.7, 0.6, 0.05)

    st.title("🧑‍🦰 42 Amman Student Face Recognition")

    uploaded_file = st.file_uploader(
        "Upload an image for face recognition", type=["jpg", "jpeg", "png"],
        help="Select a JPG, JPEG, or PNG image to detect and recognize faces"
    )

    if uploaded_file:
        try:
            image = Image.open(uploaded_file)
        except Exception:
            st.error("⚠️ Invalid image file. Please upload a valid JPG, JPEG, or PNG.")
            return

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Uploaded Image")
            st.image(image, use_container_width=True)

        @st.cache_resource
        def load_data():
            try:
                encodings = joblib.load("encodings.pkl")
                labels = joblib.load("labels.pkl")
                return encodings, labels
            except Exception as e:
                st.error(f"⚠️ Failed to load encodings or labels: {str(e)}")
                return None, None

        known_encodings, known_labels = load_data()
        if known_encodings is None or known_labels is None:
            return

        with st.spinner("Processing image..."):
            annotated_image, results = recognize_faces(image, known_encodings, known_labels, tolerance)

        with col2:
            st.subheader(f"Results — {len(results)} Face(s) Detected")
            if results:
                st.image(annotated_image, use_container_width=True)
                # Convert annotated image to bytes for download
                img_buffer = io.BytesIO()
                annotated_image.save(img_buffer, format="PNG")
                st.download_button(
                    label="Download Annotated Image",
                    data=img_buffer.getvalue(),
                    file_name="annotated_image.png",
                    mime="image/png"
                )
                for i, (label, dist) in enumerate(results, start=1):
                    if label == "Unknown":
                        st.markdown(f"**Face {i}:** ❓ Unknown — Not a 42 Amman student (Distance: {dist:.2f})")
                    else:
                        st.markdown(f"**Face {i}:** {label} ✅ — 42 Amman student (Distance: {dist:.2f})")
            else:
                st.warning("😔 No faces detected. Try a clearer image with visible faces.")

    else:
        st.info("📸 Please upload an image to begin face recognition.")

if __name__ == "__main__":
    main()
