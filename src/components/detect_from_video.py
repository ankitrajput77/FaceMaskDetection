from keras.models import load_model
import cv2, sys
from imutils.video import VideoStream
import numpy as np
from keras.applications.mobilenet_v2 import preprocess_input
from keras.preprocessing.image import img_to_array, load_img
from src.logger import logging
from src.exception import CustomException


logging.info("Model Loading Started")
# Mask Detector Model
Mask_Model = load_model('artifacts/Mask_detection_model.h5')
# Face detector models
prototxt = r"artifacts/FaceModels/deploy.prototxt"
weights = r"artifacts/FaceModels/res10_300x300_ssd_iter_140000.caffemodel"
Face_Model = cv2.dnn.readNet(prototxt, weights)
logging.info("Model Loading Ended")


# predict function 
def predict_face_mask(frame, Face_Model, Mask_Model):
	try:
		h, w = frame.shape[:2] # height n width 
		blob = cv2.dnn.blobFromImage(image=frame, 
									scalefactor=1.0, 
									size=(224, 224), 
									mean=(104.0, 177.0, 123.0)) # 4d blob
		# face detections
		Face_Model.setInput(blob)
		detections = Face_Model.forward()

		faces = list() # faces from Face Model
		locs = list() # Location of faces
		preds = list() # Prediction


		for i in range(detections.shape[2]):
			prob = detections[0, 0, i, 2]
			if prob > 0.5: # thresh for face
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(X_start, y_start, X_end, y_end) = box.astype("int")
				(X_start, y_start) = (max(0, X_start), max(0, y_start))
				(X_end, y_end) = (min(w - 1, X_end), min(h - 1, y_end))
				face = frame[y_start:y_end, X_start:X_end]
				face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
				face = cv2.resize(face, (224, 224))
				face = img_to_array(face)
				face = preprocess_input(face)
				faces.append(face)
				locs.append((X_start, y_start, X_end, y_end))
		if len(faces) > 0: # if faces
			faces = np.array(faces, dtype="float32")
			preds = Mask_Model.predict(faces, batch_size=32)
		return locs, preds
	except Exception as e:
		logging.info("Error Occured at predict_face_mask function")
		raise CustomException(e, sys)


def real_time():
	logging.info("Real time prediction started")
	# real time capture function 
	cap = VideoStream(0).start()
	while True:
		try : 
			frame = cap.read()
			locs, preds = predict_face_mask(frame, Face_Model, Mask_Model)
			# for multiple faces
			for box, pred in zip(locs, preds):
				X_start, y_start, X_end, y_end = box
				mask, withoutMask = pred

				if mask > withoutMask :
					label = "Mask"
					color = (0, 255, 0)
				else:
					label = "No Mask"
					color = (0, 0, 255)

				label = f"{label}, {format(max(mask, withoutMask) * 100, '.2f')}%"
				logging.info(label)
				cv2.putText(img=frame, text=label, org=(X_start, y_start - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.45, color=color, thickness=2, lineType=cv2.LINE_AA)
				cv2.rectangle(img=frame, pt1=(X_start, y_start), pt2=(X_end, y_end), color=color, thickness=1)

			cv2.imshow("Mask Detector", frame)
			if cv2.waitKey(2) == 27: # Esc button
				break

		except Exception as e:
			logging.info("Error at Real time Detection")
			raise CustomException(e, sys)

	cv2.destroyAllWindows()
	cap.stop()
	logging.info("All Windows Closed")