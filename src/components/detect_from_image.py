from keras.models import load_model
import cv2, sys, os
from imutils.video import VideoStream
import numpy as np
from keras.applications.mobilenet_v2 import preprocess_input
from keras.preprocessing.image import img_to_array, load_img
from src.logger import logging
from src.exception import CustomException
from PIL import Image, ImageDraw
from datetime import datetime 


logging.info("Model Loading Started")
# Mask Detector Model
Mask_Model = load_model('artifacts/Mask_detection_model.h5')
# Face detector models
prototxt = r"artifacts/FaceModels/deploy.prototxt"
weights = r"artifacts/FaceModels/res10_300x300_ssd_iter_140000.caffemodel"
Face_Model = cv2.dnn.readNet(prototxt, weights)
logging.info("Model Loading Ended")


def from_image(img):
	#img = load_img(img)
	img = img_to_array(img)
	h,w = img.shape[:2]
	blob = cv2.dnn.blobFromImage(image=img, 
										scalefactor=1.0, 
										size=(224, 224), 
										mean=(104.0, 177.0, 123.0)) # 4d blob
	Face_Model.setInput(blob)
	detections = Face_Model.forward()
	faces = []
	locs = []
	preds = []
	for i in range(detections.shape[2]):
		prob = detections[0, 0, i, 2]
		if prob > 0.5:  # thresh for face
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(X_start, y_start, X_end, y_end) = box.astype("int")
			(X_start, y_start) = (max(0, X_start), max(0, y_start))
			(X_end, y_end) = (min(w - 1, X_end), min(h - 1, y_end))
			face = img[y_start:y_end, X_start:X_end]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)
			faces.append(face)
			locs.append((X_start, y_start, X_end, y_end))
	if len(faces) > 0:  # if faces
		faces = np.array(faces, dtype="float32")
		preds = Mask_Model.predict(faces, batch_size=32)
	
	for box, pred in zip(locs, preds):
		X_start, y_start, X_end, y_end = box
		mask, withoutMask = pred
		if mask > withoutMask:
			label = "Mask"
			color = (0, 255, 0) # green
		else:
			label = "No Mask"
			color = (0, 0, 255) # red
	return (locs, preds)


def draw_rectangle(img_path, locs, preds):
    img = Image.open(img_path)
    img = img.copy()

    draw = ImageDraw.Draw(img)
    for i, face_loc in enumerate(locs):
        top_left = (face_loc[0], face_loc[1])
        bottom_right = (face_loc[2], face_loc[3])
        if preds[i][0] < preds[i][1]:
            color = 'red'
            text = 'No Mask'
        else:
            color = 'blue'
            text = 'Mask'
        draw.rectangle([top_left, bottom_right], outline=color, width=2)
        text_width, text_height = draw.textsize(text)
        x_center = (face_loc[0] + face_loc[2]) // 2
        y_top = face_loc[1] - text_height - 5
        draw.text((x_center - text_width // 2, y_top), text, fill='white')
	
    os.makedirs("savedImages",exist_ok=True)
    save_path = 'savedImages/img' + datetime.now().strftime('%d_%m_%Y_%H_%M_%S') + '.jpg'
    img.save(save_path)
    return save_path