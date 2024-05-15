# import os
# import cv2
# from paddleocr import PaddleOCR
# import numpy as np
# import imutils
# import time
# import json
# from ultralytics import YOLO
# import threading
# os.environ['KMP_DUPLICATE_LIB_OK']='True'  #for openMP error which was coming

# # Function to process each image
# def process_image(image_path, frames_dir, cropped_dir, results_file, ImgFileName, frame_nmr, json_data, license_plate_detector, lock):
#     try:
#         # Read the image
#         frame = cv2.imread(image_path)

#         # Detect license plates
#         license_plates = license_plate_detector(frame)[0]

#         for license_plate in license_plates.boxes.data.tolist():
#             x1, y1, x2, y2, lsscore, class_id = license_plate

#             # Crop license plate
#             license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2)]

#             img = license_plate_crop

#             gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#             gray = cv2.bilateralFilter(gray, 13, 15, 15)

#             edged = cv2.Canny(gray, 30, 200)

#             contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#             contours = imutils.grab_contours(contours)
#             contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

#             screenCnt = None
#             for c in contours:
#                 peri = cv2.arcLength(c, True)
#                 approx = cv2.approxPolyDP(c, 0.018 * peri, True)
#                 if len(approx) == 4:
#                     screenCnt = approx
#                     break

#             if screenCnt is not None:
#                 detected = 1
#                 cv2.drawContours(img, [screenCnt], -1, (0, 0, 255), 3)

#                 mask = np.zeros(gray.shape, np.uint8)
#                 new_image = cv2.drawContours(mask, [screenCnt], 0, 255, -1)
#                 new_image = cv2.bitwise_and(img, img, mask=mask)

#                 (x, y) = np.where(mask == 255)
#                 (topx, topy) = (np.min(x), np.min(y))
#                 (bottomx, bottomy) = (np.max(x), np.max(y))
#                 Cropped = gray[topx:bottomx + 1, topy:bottomy + 1]

#                 license_plates = ocr.ocr(Cropped, cls=True)

#                 # Save OCR results to a text file
#                 with open(results_file, 'a') as f:
#                     for result in license_plates:
#                         value = result[0][1][0]
#                         if len(result) > 0:
#                             f.write(f"{value}\n")
#                             # Save the frame
#                             with lock:
#                                 frame_path = os.path.join(frames_dir, f"{time.strftime('%d-%m-%Y_%H-%M-%S')}_{frame_nmr:05d}.jpg")
#                                 cv2.imwrite(frame_path, frame)
                    
#                             # Save the cropped lp
#                             with lock:
#                                 cropped_path = os.path.join(cropped_dir, f"{time.strftime('%d-%m-%Y_%H-%M-%S')}_{frame_nmr:05d}.jpg")
#                                 cv2.imwrite(cropped_path, license_plate_crop)

#                             #  Save the name of the file for comparision
#                             with lock:
#                                 with open(ImgFileName, 'a') as f:
#                                     f.write(f"{image_path}\n")
#     except Exception as e:
#         print(f"Error processing image: {image_path}, {e}")

# # Function to process all images in the screenshots folder
# def process_images(screenshots_folder, lock, frames_dir, cropped_dir, results_file, ImgFileName, frame_nmr, json_data, license_plate_detector):
#     # Iterate through all images in the screenshots folder
#     for filename in os.listdir(screenshots_folder):
#         # Check if the file is an image
#         if filename.endswith((".jpg", ".jpeg", ".png", ".bmp")):
#             # Construct the full filepath for the image
#             image_path = os.path.join(screenshots_folder, filename)
#             # Process the image
#             process_image(image_path, frames_dir, cropped_dir, results_file, ImgFileName, frame_nmr, json_data, license_plate_detector, lock)
#             frame_nmr += 1

# # Path to the screenshots folder
# screenshots_folder = "screenshots"

# # Create directory to store frames if not exists
# frames_dir = 'frames'
# if not os.path.exists(frames_dir):
#     os.makedirs(frames_dir)

# # Create directory to store cropped lp if not exists
# cropped_dir = 'cropped-lp'
# if not os.path.exists(cropped_dir):
#     os.makedirs(cropped_dir)

# # Parameters
# memory_decay = 1  # Memory decay in X seconds

# # Initialize variables for memory decay
# last_detection_time = {}

# # Initialize JSON data list
# json_data = []

# # Variable to keep track of frame number
# frame_nmr = -1

# # Create a lock for shared resources
# lock = threading.Lock()

# # Initialize YOLO license plate detector
# license_plate_detector = YOLO("./model/o-best.pt")

# #intiliaze the paddleOCR model
# ocr = PaddleOCR(use_angle_cls=True, lang='en')

# results_file = 'ocr_results.txt'
# ImgFileName = 'ocrImgFileNames.txt'

# # Process images in the screenshots folder
# process_images(screenshots_folder, lock, frames_dir, cropped_dir, results_file, ImgFileName, frame_nmr, json_data, license_plate_detector)

# print("Processing completed.")

# import os
# import cv2
# from paddleocr import PaddleOCR
# import numpy as np
# import imutils
# import time
# import json
# from ultralytics import YOLO
# import threading
# os.environ['KMP_DUPLICATE_LIB_OK']='True'  #for openMP error which was coming

# # Function to apply different preprocessing parameters and extract text using OCR
# def apply_preprocessing_and_ocr(image):
#     results = []
#     for bilateral_param in [(13, 15, 15)]:  # Varying parameters for bilateral filter
#         for canny_param in [(30, 200)]:  # Varying parameters for Canny edge detection
#             # Apply bilateral filter
#             gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#             gray = cv2.bilateralFilter(gray, bilateral_param[0], bilateral_param[1], bilateral_param[2])
            
#             # Apply Canny edge detection
#             edged = cv2.Canny(gray, canny_param[0], canny_param[1])

#             contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#             contours = imutils.grab_contours(contours)
#             contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

#             screenCnt = None
#             for c in contours:
#                 peri = cv2.arcLength(c, True)
#                 approx = cv2.approxPolyDP(c, 0.018 * peri, True)
#                 if len(approx) == 4:
#                     screenCnt = approx
#                     break

#             if screenCnt is not None:
#                 detected = 1
#                 cv2.drawContours(image, [screenCnt], -1, (0, 0, 255), 3)

#                 mask = np.zeros(gray.shape, np.uint8)
#                 new_image = cv2.drawContours(mask, [screenCnt], 0, 255, -1)
#                 new_image = cv2.bitwise_and(image, image, mask=mask)

#                 (x, y) = np.where(mask == 255)
#                 (topx, topy) = (np.min(x), np.min(y))
#                 (bottomx, bottomy) = (np.max(x), np.max(y))
#                 Cropped = gray[topx:bottomx + 1, topy:bottomy + 1]
            
#                 # Perform OCR
#                 ocr_results = ocr.ocr(edged, cls=True)
#                 with open(main_result_file, 'a') as f:
#                     for result in ocr_results:
#                         value = result[0][1][0]
#                         score = result[0][1][1]
#                         if len(result) > 0:
#                             f.write(f"{value} (Score: {score})\n")
#                 results.append((ocr_results, bilateral_param, canny_param))
#     return results

# # Function to process each image
# def process_image(image_path, frames_dir, cropped_dir, main_result_file, ImgFileName, otherResultFile, main_res_acc_file, frame_nmr, json_data, license_plate_detector, lock):
#     try:
#         # Read the image
#         frame = cv2.imread(image_path)

#         # Detect license plates
#         license_plates = license_plate_detector(frame)[0]

#         for license_plate in license_plates.boxes.data.tolist():
#             x1, y1, x2, y2, lsscore, class_id = license_plate

#             # Crop license plate
#             license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2)]

#             ocr_results = apply_preprocessing_and_ocr(license_plate_crop)

#             # # Save OCR results to a text file
#             # with open(main_res_acc_file, 'a') as f:
#             #     for result in ocr_results[0][0]:
#             #         value = result[0][1][0]
#             #         score = result[0][1][1]
#             #         if len(result) > 0:
#             #             f.write(f"{value} (Score: {score})\n")
#             #             # Save the frame
#             #             with lock:
#             #                 frame_path = os.path.join(frames_dir, f"{time.strftime('%d-%m-%Y_%H-%M-%S')}_{frame_nmr:05d}.jpg")
#             #                 cv2.imwrite(frame_path, frame)
                
#             #             # Save the cropped lp
#             #             with lock:
#             #                 cropped_path = os.path.join(cropped_dir, f"{time.strftime('%d-%m-%Y_%H-%M-%S')}_{frame_nmr:05d}.jpg")
#             #                 cv2.imwrite(cropped_path, license_plate_crop)

#             #             #  Save the name of the file for comparision
#             #             with lock:
#             #                  with open(ImgFileName, 'a') as f:
#             #                     f.write(f"{image_path}\n")
            
#             # # Save other OCR results to another text file
#             # with open(otherResultFile, 'a') as f:
#             #     for other_result in ocr_results[1:]:
#             #         for result in other_result[0][0]:
#             #             value = result[0][1][0]
#             #             score = result[0][1][1]
#             #             if len(result) > 0:
#             #                 f.write(f"{value} (Score: {score}) - Bilateral: {other_result[1]}, Canny: {other_result[2]}\n")

#             # Save OCR results to a text file
#             # with open(main_result_file, 'a') as f:
#             #     for result in ocr_results[0][0]:
#             #         value = result[0][1][0]
#             #         score = result[0][1][1]
#             #         if len(result) > 0:
#             #             f.write(f"{value} (Score: {score})\n")
#     except Exception as e:
#         print(f"Error processing image: {image_path}, {e}")

# # Function to process all images in the screenshots folder
# def process_images(screenshots_folder, lock, frames_dir, cropped_dir, main_result_file, ImgFileName, otherResultFile, main_res_acc_file, frame_nmr, json_data, license_plate_detector):
#     # Iterate through all images in the screenshots folder
#     for filename in os.listdir(screenshots_folder):
#         # Check if the file is an image
#         if filename.endswith((".jpg", ".jpeg", ".png", ".bmp")):
#             # Construct the full filepath for the image
#             image_path = os.path.join(screenshots_folder, filename)
#             # Process the image
#             process_image(image_path, frames_dir, cropped_dir, main_result_file, ImgFileName, otherResultFile, main_res_acc_file, frame_nmr, json_data, license_plate_detector, lock)
#             frame_nmr += 1

# # Path to the screenshots folder
# screenshots_folder = "screenshots"

# # Create directory to store frames if not exists
# frames_dir = 'frames'
# if not os.path.exists(frames_dir):
#     os.makedirs(frames_dir)

# # Create directory to store cropped lp if not exists
# cropped_dir = 'cropped-lp'
# if not os.path.exists(cropped_dir):
#     os.makedirs(cropped_dir)

# # Parameters
# memory_decay = 1  # Memory decay in X seconds

# # Initialize variables for memory decay
# last_detection_time = {}

# # Initialize JSON data list
# json_data = []

# # Variable to keep track of frame number
# frame_nmr = -1

# # Create a lock for shared resources
# lock = threading.Lock()

# # Initialize YOLO license plate detector
# license_plate_detector = YOLO("./model/o-best.pt")

# #intiliaze the paddleOCR model
# ocr = PaddleOCR(use_angle_cls=True, lang='en')

# main_result_file = 'main_ocr_results.txt'
# ImgFileName = 'ocrImgFileNames.txt'
# otherResultFile = 'ocrOtherResults.txt'
# main_res_acc_file = 'res_acc_file.txt'

# # Process images in the screenshots folder
# process_images(screenshots_folder, lock, frames_dir, cropped_dir, main_result_file, ImgFileName, otherResultFile, main_res_acc_file, frame_nmr, json_data, license_plate_detector)

# print("Processing completed.")


# ****************** MY OWN EXPERIMENT ***********************


# import os
# import cv2
# from paddleocr import PaddleOCR
# import numpy as np
# import imutils
# import time
# import json
# from ultralytics import YOLO
# import threading
# os.environ['KMP_DUPLICATE_LIB_OK']='True'  #for openMP error which was coming

# # Function to apply different preprocessing parameters and extract text using OCR
# def apply_preprocessing_and_ocr(image, count, image_path, frame, main_result_file):
#     for bilateral_param in [(13, 15, 15)]:  # Varying parameters for bilateral filter
#         for canny_param in [(30, 200)]:  # Varying parameters for Canny edge detection
#             # Apply bilateral filter
#             gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#             gray = cv2.bilateralFilter(gray, bilateral_param[0], bilateral_param[1], bilateral_param[2])
            
#             # Apply Canny edge detection
#             edged = cv2.Canny(gray, canny_param[0], canny_param[1])

#             contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#             contours = imutils.grab_contours(contours)
#             contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

#             screenCnt = None
#             for c in contours:
#                 peri = cv2.arcLength(c, True)
#                 approx = cv2.approxPolyDP(c, 0.018 * peri, True)
#                 if len(approx) == 4:
#                     screenCnt = approx
#                     break

#             if screenCnt is not None:
#                 detected = 1
#                 cv2.drawContours(image, [screenCnt], -1, (0, 0, 255), 3)

#                 mask = np.zeros(gray.shape, np.uint8)
#                 new_image = cv2.drawContours(mask, [screenCnt], 0, 255, -1)
#                 new_image = cv2.bitwise_and(image, image, mask=mask)

#                 (x, y) = np.where(mask == 255)
#                 (topx, topy) = (np.min(x), np.min(y))
#                 (bottomx, bottomy) = (np.max(x), np.max(y))
#                 Cropped = gray[topx:bottomx + 1, topy:bottomy + 1]
            
#                 # Perform OCR
#                 ocr_results = ocr.ocr(Cropped, cls=True)

#                 #file operations performed here
#                 # This is for the 1st choice, that is our main result
#                 if (count == 0):
#                     count = count + 1
#                     with open(main_result_file, 'a') as f:
#                         for result in ocr_results:
#                             value = result[0][1][0]
#                             score = result[0][1][1]
#                             if len(result) > 0:
#                                 f.write(f"{value} (Score: {score})\n")
#                                 # Save the frame in frame dircetory
#                                 with lock:
#                                     frame_path = os.path.join(frames_dir, f"{time.strftime('%d-%m-%Y_%H-%M-%S')}_{frame_nmr:05d}.jpg")
#                                     cv2.imwrite(frame_path, frame)
                
#                                 # Save the cropped lp
#                                 with lock:
#                                     cropped_path = os.path.join(cropped_dir, f"{time.strftime('%d-%m-%Y_%H-%M-%S')}_{frame_nmr:05d}.jpg")
#                                     cv2.imwrite(cropped_path, image) #image is the cropped license plate

#                                 #  Save the name of the file for comparision
#                                 with lock:
#                                     with open(ImgFileName, 'a') as f:
#                                         f.write(f"{image_path}\n")

#                 # if its not the 0th elements, means these results are our secondary choice
#                 else:
#                     # Save other OCR results to another text file
#                     with open(otherResultFile, 'a') as f:
#                         for result in ocr_results:
#                             value = result[0][1][0]
#                             score = result[0][1][1]
#                             if len(result) > 0:
#                                 f.write(f"{value} (Score: {score}) - Bilateral: {bilateral_param[0]}, {bilateral_param[1]}, {bilateral_param[2]}, Canny: {canny_param[0]}, {canny_param[1]}\n")

            
#         count = count + 1

# # Function to process each image
# def process_image(image_path, frames_dir, cropped_dir, main_result_file, ImgFileName, otherResultFile, main_res_acc_file, frame_nmr, json_data, license_plate_detector, lock):
#     try:
#         # Read the image
#         frame = cv2.imread(image_path)

#         # Detect license plates
#         license_plates = license_plate_detector(frame)[0]

#         for license_plate in license_plates.boxes.data.tolist():
#             x1, y1, x2, y2, lsscore, class_id = license_plate
#             count = 0

#             # Crop license plate
#             license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2)]

#             apply_preprocessing_and_ocr(license_plate_crop, count, image_path, frame, main_result_file)
#     except Exception as e:
#         print(f"Error processing image: {image_path}, {e}")

# # Function to process all images in the screenshots folder
# def process_images(screenshots_folder, lock, frames_dir, cropped_dir, main_result_file, ImgFileName, otherResultFile, main_res_acc_file, frame_nmr, json_data, license_plate_detector):
#     # Iterate through all images in the screenshots folder
#     for filename in os.listdir(screenshots_folder):
#         # Check if the file is an image
#         if filename.endswith((".jpg", ".jpeg", ".png", ".bmp")):
#             # Construct the full filepath for the image
#             image_path = os.path.join(screenshots_folder, filename)
#             # Process the image
#             process_image(image_path, frames_dir, cropped_dir, main_result_file, ImgFileName, otherResultFile, main_res_acc_file, frame_nmr, json_data, license_plate_detector, lock)
#             frame_nmr += 1

# # Path to the screenshots folder
# screenshots_folder = "screenshots"

# # Create directory to store frames if not exists
# frames_dir = 'frames'
# if not os.path.exists(frames_dir):
#     os.makedirs(frames_dir)

# # Create directory to store cropped lp if not exists
# cropped_dir = 'cropped-lp'
# if not os.path.exists(cropped_dir):
#     os.makedirs(cropped_dir)

# # Parameters
# memory_decay = 1  # Memory decay in X seconds

# # Initialize variables for memory decay
# last_detection_time = {}

# # Initialize JSON data list
# json_data = []

# # Variable to keep track of frame number
# frame_nmr = -1

# # Create a lock for shared resources
# lock = threading.Lock()

# # Initialize YOLO license plate detector
# license_plate_detector = YOLO("./model/o-best.pt")

# #intiliaze the paddleOCR model
# ocr = PaddleOCR(use_angle_cls=True, lang='en')

# main_result_file = 'main_ocr_results.txt'
# ImgFileName = 'ocrImgFileNames.txt'
# otherResultFile = 'ocrOtherResults.txt'
# main_res_acc_file = 'res_acc_file.txt'

# # Process images in the screenshots folder
# process_images(screenshots_folder, lock, frames_dir, cropped_dir, main_result_file, ImgFileName, otherResultFile, main_res_acc_file, frame_nmr, json_data, license_plate_detector)

# print("Processing completed.")


# ********************************** My logic of two arrays and accessing them ***************************************


# import os
# import cv2
# from paddleocr import PaddleOCR
# import numpy as np
# import imutils
# import time
# import json
# from ultralytics import YOLO
# import threading
# os.environ['KMP_DUPLICATE_LIB_OK']='True'  #for openMP error which was coming

# # Function to apply different preprocessing parameters and extract text using OCR
# def apply_preprocessing_and_ocr(image, image_path, frame, main_result_file, otherResultFile):

#     bilateral_array = [(13,15,15), (13,15,15), (13,15,15), (13,15,15)]
#     canny_array = [(30,200), (30,200), (30,200), (30,200)]

#     for i in range (len(bilateral_array)):
#         print("Checking iterations")
#         print(i)
#         current_bf = bilateral_array[i]
#         current_can = canny_array[i]

#         # Apply bilateral filter
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         gray = cv2.bilateralFilter(gray, current_bf[0], current_bf[1], current_bf[2])
            
#         cv2.imwrite("img3.jpg", gray)
#         # Apply Canny edge detection
#         edged = cv2.Canny(gray, current_can[0], current_can[1])
#         cv2.imwrite("img2.jpg", edged)

#         contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#         contours = imutils.grab_contours(contours)
#         contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]


#         screenCnt = None
#         for c in contours:
#             peri = cv2.arcLength(c, True)
#             approx = cv2.approxPolyDP(c, 0.018 * peri, True)
#             print("Inside contour loop")
#             print(len(approx))
#             if len(approx) == 4:
#                 screenCnt = approx
#                 break

#         if screenCnt is not None:
#             detected = 1
#             cv2.drawContours(image, [screenCnt], -1, (0, 0, 255), 3)

#             mask = np.zeros(gray.shape, np.uint8)
#             new_image = cv2.drawContours(mask, [screenCnt], 0, 255, -1)
#             new_image = cv2.bitwise_and(image, image, mask=mask)

#             (x, y) = np.where(mask == 255)
#             (topx, topy) = (np.min(x), np.min(y))
#             (bottomx, bottomy) = (np.max(x), np.max(y))
#             Cropped = gray[topx:bottomx + 1, topy:bottomy + 1]
            
#             # Perform OCR
#             ocr_results = ocr.ocr(Cropped, cls=True)
#             time.sleep(0.5)

#             if(i == 0):
#                 print("For 0-th index")
#                 print(i)
#                 # Save OCR results to a text file
#                 with open(main_result_file, 'a') as f:
#                     for result in ocr_results:
#                         value = result[0][1][0]
#                         score = result[0][1][1]
#                         if len(result) > 0:
#                             f.write(f"{value} (Score: {score})\n")
#                             # Save the frame in frame dircetory
#                             with lock:
#                                 frame_path = os.path.join(frames_dir, f"{time.strftime('%d-%m-%Y_%H-%M-%S')}_{frame_nmr:05d}.jpg")
#                                 cv2.imwrite(frame_path, frame)
                
#                             # Save the cropped lp
#                             with lock:
#                                 cropped_path = os.path.join(cropped_dir, f"{time.strftime('%d-%m-%Y_%H-%M-%S')}_{frame_nmr:05d}.jpg")
#                                 cv2.imwrite(cropped_path, image) #image is the cropped license plate

#                             #  Save the name of the file for comparision
#                             with lock:
#                                 with open(ImgFileName, 'a') as f:
#                                     f.write(f"{image_path}\n")
#             if(i > 0):
#                 print("For other index")
#                 print(i)
#                 # Save other OCR results to another text file
#                 with open(otherResultFile, 'a') as f:
#                     print("inside file")
#                     for result in ocr_results:
#                         print("inside result for loop")
#                         print(result)
#                         value = result[0][1][0]
#                         print("inside value")
#                         score = result[0][1][1]
#                         print("inside score")
#                         if len(result) > 0:
#                             f.write(f"{value} (Score: {score}) - Bilateral: {current_bf[0]}, {current_bf[1]}, {current_bf[2]}, Canny: {current_can[0]}, {current_can[1]}\n")

# # Function to process each image
# def process_image(image_path, frames_dir, cropped_dir, main_result_file, ImgFileName, otherResultFile, main_res_acc_file, frame_nmr, json_data, license_plate_detector, lock):
#     try:
#         # Read the image
#         frame = cv2.imread(image_path)

#         # Detect license plates
#         license_plates = license_plate_detector(frame)[0]

#         for license_plate in license_plates.boxes.data.tolist():
#             x1, y1, x2, y2, lsscore, class_id = license_plate

#             # Crop license plate
#             license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2)]

#             apply_preprocessing_and_ocr(license_plate_crop, image_path, frame, main_result_file, otherResultFile)
#     except Exception as e:
#         print(f"Error processing image: {image_path}, {e}")

# # Function to process all images in the screenshots folder
# def process_images(screenshots_folder, lock, frames_dir, cropped_dir, main_result_file, ImgFileName, otherResultFile, main_res_acc_file, frame_nmr, json_data, license_plate_detector):
#     # Iterate through all images in the screenshots folder
#     for filename in os.listdir(screenshots_folder):
#         # Check if the file is an image
#         if filename.endswith((".jpg", ".jpeg", ".png", ".bmp")):
#             # Construct the full filepath for the image
#             image_path = os.path.join(screenshots_folder, filename)
#             # Process the image
#             process_image(image_path, frames_dir, cropped_dir, main_result_file, ImgFileName, otherResultFile, main_res_acc_file, frame_nmr, json_data, license_plate_detector, lock)
#             frame_nmr += 1

# # Path to the screenshots folder
# screenshots_folder = "ss"

# # Create directory to store frames if not exists
# frames_dir = 'frames'
# if not os.path.exists(frames_dir):
#     os.makedirs(frames_dir)

# # Create directory to store cropped lp if not exists
# cropped_dir = 'cropped-lp'
# if not os.path.exists(cropped_dir):
#     os.makedirs(cropped_dir)

# # Parameters
# memory_decay = 1  # Memory decay in X seconds

# # Initialize variables for memory decay
# last_detection_time = {}

# # Initialize JSON data list
# json_data = []

# # Variable to keep track of frame number
# frame_nmr = -1

# # Create a lock for shared resources
# lock = threading.Lock()

# # Initialize YOLO license plate detector
# license_plate_detector = YOLO("./model/o-best.pt")

# #intiliaze the paddleOCR model
# ocr = PaddleOCR(use_angle_cls=True, lang='en')

# main_result_file = 'main_ocr_results.txt'
# ImgFileName = 'ocrImgFileNames.txt'
# otherResultFile = 'ocrOtherResults.txt'
# main_res_acc_file = 'res_acc_file.txt'

# # Process images in the screenshots folder
# process_images(screenshots_folder, lock, frames_dir, cropped_dir, main_result_file, ImgFileName, otherResultFile, main_res_acc_file, frame_nmr, json_data, license_plate_detector)

# print("Processing completed.")


# ******************** To just test the results of ocr ith gray scaling and text extration ****************************************


import os
import cv2
from paddleocr import PaddleOCR
import numpy as np
import imutils
import time
import json
from ultralytics import YOLO
import threading
os.environ['KMP_DUPLICATE_LIB_OK']='True'  #for openMP error which was coming

# Function to apply different preprocessing parameters and extract text using OCR
def apply_preprocessing_and_ocr(image, image_path, frame, main_result_file, otherResultFile):

    bilateral_array = [(9, 45, 45), (5, 15, 15), (9, 25, 25),(11, 35, 35) , (13,15,15)]

    with open(otherResultFile, 'a') as f:
        f.write(f"********************************************************\n")

    for i in range (len(bilateral_array)):
        print("Checking iterations")
        print(i)
        current_bf = bilateral_array[i]

        # Apply bilateral filter
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, current_bf[0], current_bf[1], current_bf[2])
    
        # Perform OCR
        ocr_results = ocr.ocr(gray, cls=True)
        time.sleep(0.5)

        if(i == 0):
            print("For 0-th index")
            print(i)
            # Save OCR results to a text file
            with open(main_result_file, 'a') as f:
                for result in ocr_results:
                    value = result[0][1][0]
                    score = result[0][1][1]
                    if len(result) > 0:
                        f.write(f"{value}\n")
                        # Save the frame in frame dircetory
                        with lock:
                            frame_path = os.path.join(frames_dir, f"{time.strftime('%d-%m-%Y_%H-%M-%S')}_{frame_nmr:05d}.jpg")
                            cv2.imwrite(frame_path, frame)
                
                        # Save the cropped lp
                        with lock:
                            cropped_path = os.path.join(cropped_dir, f"{time.strftime('%d-%m-%Y_%H-%M-%S')}_{frame_nmr:05d}.jpg")
                            cv2.imwrite(cropped_path, image) #image is the cropped license plate

                        #  Save the name of the file for comparision
                        with lock:
                            with open(ImgFileName, 'a') as f:
                                f.write(f"{image_path}\n")
        if(i > 0):
            print("For other index")
            print(i)
            # Save other OCR results to another text file
            with open(otherResultFile, 'a') as f:
                print("inside file")
                for result in ocr_results:
                    print("inside result for loop")
                    print(result)
                    value = result[0][1][0]
                    print("inside value")
                    score = result[0][1][1]
                    print("inside score")
                    if len(result) > 0:
                        f.write(f"{value} (Score: {score}) - Bilateral: {current_bf[0]}, {current_bf[1]}, {current_bf[2]}\n")

# Function to process each image
def process_image(image_path, frames_dir, cropped_dir, main_result_file, ImgFileName, otherResultFile, main_res_acc_file, frame_nmr, json_data, license_plate_detector, lock):
    try:
        # Read the image
        frame = cv2.imread(image_path)

        # Detect license plates
        license_plates = license_plate_detector(frame)[0]

        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, lsscore, class_id = license_plate

            # Crop license plate
            license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2)]

            apply_preprocessing_and_ocr(license_plate_crop, image_path, frame, main_result_file, otherResultFile)
            break #only one license plate is processed per frame
    except Exception as e:
        print(f"Error processing image: {image_path}, {e}")

# Function to process all images in the screenshots folder
def process_images(screenshots_folder, lock, frames_dir, cropped_dir, main_result_file, ImgFileName, otherResultFile, main_res_acc_file, frame_nmr, json_data, license_plate_detector):
    # Iterate through all images in the screenshots folder
    for filename in os.listdir(screenshots_folder):
        # Check if the file is an image
        if filename.endswith((".jpg", ".jpeg", ".png", ".bmp")):
            # Construct the full filepath for the image
            image_path = os.path.join(screenshots_folder, filename)
            # Process the image
            process_image(image_path, frames_dir, cropped_dir, main_result_file, ImgFileName, otherResultFile, main_res_acc_file, frame_nmr, json_data, license_plate_detector, lock)
            frame_nmr += 1

# Path to the screenshots folder
screenshots_folder = "screenshots"

# Create directory to store frames if not exists
frames_dir = 'frames'
if not os.path.exists(frames_dir):
    os.makedirs(frames_dir)

# Create directory to store cropped lp if not exists
cropped_dir = 'cropped-lp'
if not os.path.exists(cropped_dir):
    os.makedirs(cropped_dir)

# Parameters
memory_decay = 1  # Memory decay in X seconds

# Initialize variables for memory decay
last_detection_time = {}

# Initialize JSON data list
json_data = []

# Variable to keep track of frame number
frame_nmr = -1

# Create a lock for shared resources
lock = threading.Lock()

# Initialize YOLO license plate detector
license_plate_detector = YOLO("./model/o-best.pt")

#intiliaze the paddleOCR model
ocr = PaddleOCR(use_angle_cls=True, lang='en')

main_result_file = 'main_ocr_results.txt'
ImgFileName = 'ocrImgFileNames.txt'
otherResultFile = 'ocrOtherResults.txt'
main_res_acc_file = 'res_acc_file.txt'

# Process images in the screenshots folder
process_images(screenshots_folder, lock, frames_dir, cropped_dir, main_result_file, ImgFileName, otherResultFile, main_res_acc_file, frame_nmr, json_data, license_plate_detector)

print("Processing completed.")