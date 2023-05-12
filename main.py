from backbone import TFLiteModel, get_model
from landmarks_extraction import mediapipe_detection, draw, extract_coordinates, load_json_file,  CFG 
from config import SEQ_LEN, THRESH_HOLD
import numpy as np
import pandas as pd
import cv2
import time
import mediapipe as mp

mp_holistic = mp.solutions.holistic 
mp_drawing = mp.solutions.drawing_utils

s2p_map = {k.lower():v for k,v in load_json_file(CFG.data_dir + "sign_to_prediction_index_map.json").items()}
p2s_map = {v:k for k,v in load_json_file(CFG.data_dir + "sign_to_prediction_index_map.json").items()}
encoder = lambda x: s2p_map.get(x.lower())
decoder = lambda x: p2s_map.get(x)

models_path = [
                '/Users/sontung/Documents /MyDocuments/NEU/Thesis/sign_language/models/islr-fp16-192-8-seed_all42-foldall-last.h5',
]
models = [get_model() for _ in models_path]
for model,path in zip(models,models_path):
    model.load_weights(path)

def real_time_asl():
    res = []
    tflite_keras_model = TFLiteModel(islr_models=models)
    sequence_data = []
    cap = cv2.VideoCapture(0)
    
    start = time.time()
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            image, results = mediapipe_detection(frame, holistic)
            draw(image, results)

            try:
                landmarks = extract_coordinates(results)
            except:
                landmarks = np.zeros((468 + 21 + 33 + 21, 3))
            sequence_data.append(landmarks)
            
            sign = ""
            
            if len(sequence_data) % SEQ_LEN == 0:
                prediction = tflite_keras_model(np.array(sequence_data, dtype = np.float32))["outputs"]

                print(np.max(prediction.numpy(), axis=-1))
                if np.max(prediction.numpy(), axis=-1) > THRESH_HOLD:
                    sign = np.argmax(prediction.numpy(), axis=-1)
                
                sequence_data = []
            
            cv2.putText(image, f"{len(sequence_data)}", (3, 35),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            
            # print(len(sequence_data))
            if sign != "" and decoder(sign) not in res:
                res.insert(0, decoder(sign))
            
            # Get the height and width of the image
            height, width = image.shape[0], image.shape[1]

            # Create a white column
            white_column = np.ones((height // 8, width, 3), dtype='uint8') * 255

            # Concatenate the white column to the image
            image = np.concatenate((white_column, image), axis=0)
            
            cv2.putText(image, f"{', '.join(str(x) for x in res)}", (3, 65),
                                cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 2, cv2.LINE_AA)
                            
            cv2.imshow('Webcam Feed',image)
            if cv2.waitKey(10) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

real_time_asl()