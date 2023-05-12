from backbone import TFLiteModel, get_model
from utils import read_json_file, load_relevant_data_subset

import tensorflow as tf
import numpy as np
import pandas as pd
import json
import cv2
import os
import time
import mediapipe as mp

from multiprocessing import cpu_count


mp_holistic = mp.solutions.holistic # holistic model
mp_drawing = mp.solutions.drawing_utils # drawing utilities

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # color conversion
    image.flags.writeable = False # img no longer writeable
    pred = model.process(image) # make landmark prediction
    image.flags.writeable = True  # img now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # color reconversion
    return image, pred

def draw(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                              mp_drawing.DrawingSpec(color=(0,0,0), thickness=1, circle_radius=0),
                              mp_drawing.DrawingSpec(color=(227, 224, 113), thickness=1, circle_radius=0))
    # mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
    #                           mp_drawing.DrawingSpec(color=(0,150,0), thickness=3, circle_radius=3),
    #                           mp_drawing.DrawingSpec(color=(0,0,0), thickness=2, circle_radius=2))
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(200,56,12), thickness=3, circle_radius=3),
                              mp_drawing.DrawingSpec(color=(0,0,0), thickness=2, circle_radius=2))
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(250,56,12), thickness=3, circle_radius=3),
                              mp_drawing.DrawingSpec(color=(0,0,0), thickness=2, circle_radius=2))
    
def extract_coordinates(results):
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]) if results.face_landmarks.landmark else np.zeros((468, 3))
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]) if results.pose_landmarks.landmark else np.zeros((33, 3))
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]) if results.left_hand_landmarks.landmark else np.zeros((21, 3))
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]) if results.right_hand_landmarks.landmark else np.zeros((21, 3))
    return np.concatenate([face, lh, pose, rh])
    
def load_json_file(json_path):
    with open(json_path, 'r') as f:
        sign_map = json.load(f)
    return sign_map

class CFG:
    data_dir = "/Users/sontung/Documents /MyDocuments/NEU/Thesis/sign_language/resources/"
    sequence_length = 12
    rows_per_frame = 543

ROWS_PER_FRAME = 543
def load_relevant_data_subset(pq_path):
    data_columns = ['x', 'y', 'z']
    data = pd.read_parquet(pq_path, columns=data_columns)
    n_frames = int(len(data) / ROWS_PER_FRAME)
    data = data.values.reshape(n_frames, ROWS_PER_FRAME, len(data_columns))
    return data.astype(np.float32)
    
sign_map = load_json_file(CFG.data_dir + 'sign_to_prediction_index_map.json')
train_data = pd.read_csv(CFG.data_dir + 'train.csv')

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
            
            
            if len(sequence_data) % 30 == 0:
                # print(time.time() - start)
                prediction = tflite_keras_model(np.array(sequence_data, dtype = np.float32))["outputs"]

                print(np.max(prediction.numpy(), axis=-1))
                if np.max(prediction.numpy(), axis=-1) > 0.2:
                    sign = np.argmax(prediction.numpy(), axis=-1)
                
                sequence_data = []
            
            cv2.putText(image, f"{len(sequence_data)}", (3, 35),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            
            # print(len(sequence_data))
            if sign != "" and decoder(sign) not in res:
                res.insert(0, decoder(sign))
            
            
            # cv2.putText(image, f"Prediction:    {decoder(sign)}", (3, 35),
            #                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
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