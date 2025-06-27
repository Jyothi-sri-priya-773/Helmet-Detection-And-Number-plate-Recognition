import torch
import cv2
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import io
import numpy as np
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# Load the pre-trained YOLOv8 model from the saved weights
model_path = r'runs\detect\train\weights\best.pt'  # Path to your YOLOv8 model weights
model = YOLO(model_path)  # Load YOLOv8 model

# # Set the path for Tesseract executable
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load the TrOCR processor and model
ocr_processor = TrOCRProcessor.from_pretrained("PawanKrGunjan/license_plate_recognizer")
ocr_model = VisionEncoderDecoderModel.from_pretrained("PawanKrGunjan/license_plate_recognizer")

# Function to detect objects using YOLO and crop relevant regions
def detect_and_crop_parts(image_path, model):
    # Read image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Perform inference using YOLOv8
    results = model(img_rgb)  # Perform inference

    # Extract the first result (if we are working with one image)
    result = results[0]

    # Get bounding boxes, class IDs, and labels
    boxes = result.boxes
    class_ids = boxes.cls  # Accessing class IDs for each detection
    class_labels = result.names  # Accessing class names from model

    if len(boxes) == 0:
        print("No objects detected.")
        return img, []

    cropped_parts = []

    # Iterate through the detected boxes and process each detected object
    for i, box in enumerate(boxes):
        # Extract bounding box coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Get the class label for the detected object
        class_label = class_labels[int(class_ids[i])]

        # Draw the bounding box around the detected object
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
        cv2.putText(img, class_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Crop the object region
        part_img = img[y1:y2, x1:x2]  # Crop the detected region
        cropped_parts.append((class_label, part_img))

    return img, cropped_parts

# Function to extract text from the cropped license plate image using Tesseract and TrOCR
def extract_text_from_plate(cropped_parts, processor, ocr_model):
    detected_texts = []

    for class_label, part_img in cropped_parts:
        if class_label != "number plate":
            continue  # Skip non-license plate parts for OCR

        # Convert cropped image to grayscale for better OCR accuracy
        gray_plate = cv2.cvtColor(part_img, cv2.COLOR_BGR2GRAY)

        # Apply denoising (optional but can help with noisy images)
        gray_plate = cv2.fastNlMeansDenoising(gray_plate, None, 30, 7, 21)

        # Apply adaptive thresholding (helps with varying light conditions)
        gray_plate = cv2.adaptiveThreshold(gray_plate, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                           cv2.THRESH_BINARY, 11, 2)

        # Apply OCR to the cropped license plate region using Tesseract
        # text_tesseract = pytesseract.image_to_string(gray_plate, config='--psm 7')

        # # Log raw OCR output for debugging
        # print(f"Tesseract OCR output: '{text_tesseract}'")

        # if text_tesseract.strip():
        #     detected_texts.append(text_tesseract.strip())

        # Use TrOCR to extract the number plate text
        pil_img = Image.fromarray(gray_plate)  # Convert to PIL image
        pil_img_rgb = pil_img.convert("RGB")  # Ensure it's in RGB format
        pixel_values = processor(pil_img_rgb, return_tensors="pt").pixel_values
        generated_ids = ocr_model.generate(pixel_values)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        print(f"TrOCR output: '{generated_text}'")

        if generated_text.strip():
            detected_texts.append(generated_text.strip())

    return detected_texts

# Pages
PAGES = {
    "Home": "home",
    "Register": "register",
    "Login": "login",
    "Upload": "upload",
}

# State for login
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# Streamlit navigation
def home_page():
    st.title("Welcome to the Object Detection and OCR App")
    if st.session_state.logged_in:
        st.success("You are logged in. Navigate to the Upload page to start!")
    else:
        st.write("Navigate using the sidebar to Register, Login, or Upload images.")

def register_page():
    st.title("Register")
    username = st.text_input("Enter a username:")
    password = st.text_input("Enter a password:", type="password")
    confirm_password = st.text_input("Confirm your password:", type="password")

    if st.button("Register"):
        if password != confirm_password:
            st.error("Passwords do not match.")
        elif len(username) == 0 or len(password) == 0:
            st.error("Username and password cannot be empty.")
        else:
            # Simulate saving user credentials (for example purposes only)
            st.session_state["user_cred"] = {"username": username, "password": password}
            st.success("Registration successful! Proceed to the Login page.")

def login_page():
    st.title("Login")
    username = st.text_input("Username:")
    password = st.text_input("Password:", type="password")

    if st.button("Login"):
        saved_cred = st.session_state.get("user_cred", {})
        if saved_cred.get("username") == username and saved_cred.get("password") == password:
            st.session_state.logged_in = True
            st.success("Login successful! Navigate to the Upload page.")
        else:
            st.error("Invalid username or password.")

def upload_page():
    if not st.session_state.logged_in:
        st.error("You must be logged in to access this page.")
        return

    st.title("Object Detection and OCR")
    st.write("Upload an image to detect objects and extract text from license plates.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Convert the uploaded file to a format that OpenCV can use
        image_bytes = uploaded_file.read()
        image_stream = io.BytesIO(image_bytes)

        # Open the image using PIL to ensure it's a valid image
        try:
            image = Image.open(image_stream)
            image = image.convert('RGB')  # Ensure it's in RGB format
        except IOError:
            st.error("The uploaded file is not a valid image.")
            st.stop()

        # Save the image to a temporary path for OpenCV processing
        image_path = "temp_image.jpg"
        image.save(image_path)  # Save as JPEG format to be read by OpenCV

        # Call the function to detect and crop the parts
        img_with_boxes, cropped_parts = detect_and_crop_parts(image_path, model)

        # Display the processed image with bounding boxes
        if img_with_boxes is not None:
            st.image(cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB), caption="Processed Image with Bounding Boxes", use_container_width=True)

        # Call the function to extract text from the cropped license plates
        if cropped_parts:
            detected_plates = extract_text_from_plate(cropped_parts, ocr_processor, ocr_model)

            # Display the extracted plate text for each detected plate
            st.write("Detected plate text:")
            for plate_text in detected_plates:
                st.write(f"- {plate_text}")
        else:
            st.write("No valid objects detected.")

# Logout function
def logout():
    st.session_state.logged_in = False
    st.sidebar.success("You have been logged out.")

# Sidebar navigation
st.sidebar.title("Navigation")
if st.session_state.logged_in:
    st.sidebar.button("Logout", on_click=logout)

selection = st.sidebar.radio("Go to", list(PAGES.keys()))

if selection == "Home":
    home_page()
elif selection == "Register":
    register_page()
elif selection == "Login":
    login_page()
elif selection == "Upload":
    upload_page()
