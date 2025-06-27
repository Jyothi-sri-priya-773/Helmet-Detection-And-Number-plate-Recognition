import sqlite3
import torch
import cv2
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import io
import numpy as np
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import time

# Model paths and loading
@st.cache_resource
def load_models():
    ocr_processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-printed')
    ocr_model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-printed')
    
    model_path = r'best.pt'
    model = YOLO(model_path)  # Load YOLO model
    return ocr_processor, ocr_model, model

ocr_processor, ocr_model, model = load_models()

# Function to detect and crop parts from the image
def detect_and_crop_parts(image, model):
    img = np.array(image)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    results = model(img_rgb)  # Use model here
    result = results[0]

    boxes = result.boxes
    class_ids = boxes.cls
    class_labels = result.names
    if len(boxes) == 0:
        return img, []

    cropped_parts = []

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        class_label = class_labels[int(class_ids[i])]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, class_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        part_img = img[y1:y2, x1:x2]  # Crop the detected region
        cropped_parts.append((class_label, part_img))

    return img, cropped_parts

# Function to extract text using TrOCR
def extract_text_from_plate(cropped_parts, processor, ocr_model):
    detected_texts = []

    for class_label, part_img in cropped_parts:
        if class_label != "number plate":
            continue  # Skip non-license plate parts for OCR
        gray_plate = cv2.cvtColor(part_img, cv2.COLOR_BGR2GRAY)

        # Apply adaptive thresholding with varying light conditions
        gray_plate = cv2.adaptiveThreshold(gray_plate, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                           cv2.THRESH_BINARY, 11, 2)

        pil_img = Image.fromarray(gray_plate)  
        pil_img_rgb = pil_img.convert("RGB") 
        pixel_values = processor(pil_img_rgb, return_tensors="pt").pixel_values
        generated_ids = ocr_model.generate(pixel_values)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        if generated_text.strip():
            detected_texts.append(generated_text.strip())

    return detected_texts

# Streamlit navigation
def home_page():
    st.title("Welcome to the Object Detection and OCR App")
    st.write("Upload an image to detect objects and extract text from license plates.")

def upload_page():
    st.title("Object Detection and OCR")
    st.write("Upload an image to detect objects and extract text from license plates.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image_bytes = uploaded_file.read()
        image_stream = io.BytesIO(image_bytes)
        try:
            image = Image.open(image_stream)
            image = image.convert('RGB')  # Convert to RGB format to ensure correct processing
        except IOError:
            st.error("The uploaded file is not a valid image.")
            st.stop()

        # Resize image for faster processing (if it's too large)
        base_width = 800
        w_percent = (base_width / float(image.size[0]))
        h_size = int((float(image.size[1]) * float(w_percent)))
        image = image.resize((base_width, h_size), Image.Resampling.LANCZOS)


        start_time = time.time()  # Measure time taken for detection and OCR
        img_with_boxes, cropped_parts = detect_and_crop_parts(image, model)  # Use model here
        
        if img_with_boxes is not None:
            st.image(cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB), caption="Processed Image with Bounding Boxes", use_container_width=True)
        
        if cropped_parts:
            detected_plates = extract_text_from_plate(cropped_parts, ocr_processor, ocr_model)  # Pass processor and model here
            st.write("Detected plate text:")
            for plate_text in detected_plates:
                st.write(f"- {plate_text}")
        
        processing_time = time.time() - start_time  # Display processing time
        st.write(f"Processing time: {processing_time:.2f} seconds")
        
        if not cropped_parts:
            st.write("No valid objects detected.")

# Sidebar navigation
st.sidebar.title("Navigation")

selection = st.sidebar.radio("Go to", ["Home", "Upload"])

if selection == "Home":
    home_page()
elif selection == "Upload":
    upload_page()
