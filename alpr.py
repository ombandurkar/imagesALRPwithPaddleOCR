import os
import cv2
from paddleocr import PaddleOCR

# Function to process each image
def process_image(image_path, ocr, frames_dir, cropped_dir, results_file):
    try:
        # Read the image
        frame = cv2.imread(image_path)

        if frame is not None:
            # Perform license plate detection and OCR
            license_plates = ocr.ocr(frame, cls=True)

            # Save OCR results to a text file
            with open(results_file, 'a') as f:
                for result in license_plates:
                    value = result[0][1][0]
                    f.write(f"{value}\n")

            # Save frames and cropped license plate images
            frame_name = os.path.basename(image_path)
            frame_output_path = os.path.join(frames_dir, frame_name)
            cv2.imwrite(frame_output_path, frame)

            for idx, result in enumerate(license_plates):
                cropped_image = frame[result[0][1]:result[0][3], result[0][0]:result[0][2]]
                cv2.imwrite(cropped_image)
        else:
            print(f"Error reading image: {image_path}")
    except Exception as e:
        print(f"Error processing image: {image_path}, {e}")

# Function to process all images in the screenshots folder
def process_images(screenshots_folder, ocr, frames_dir, cropped_dir, results_file):
    # Create directories if not exists
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)
    if not os.path.exists(cropped_dir):
        os.makedirs(cropped_dir)

    # Iterate through all images in the screenshots folder
    for filename in os.listdir(screenshots_folder):
        # Check if the file is an image
        if filename.endswith((".jpg", ".jpeg", ".png", ".bmp")):
            # Construct the full filepath for the image
            image_path = os.path.join(screenshots_folder, filename)
            # Process the image
            process_image(image_path, ocr, frames_dir, cropped_dir, results_file)

# Path to the screenshots folder
screenshots_folder = "screenshots"

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Paths for output directories and results file
frames_dir = 'frames'
cropped_dir = 'cropped-lp'
results_file = 'ocr_results.txt'

# Process images in the screenshots folder
process_images(screenshots_folder, ocr, frames_dir, cropped_dir, results_file)

print("Processing completed.")