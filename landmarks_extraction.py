from config import ROWS_PER_FRAME, DATA_DIR, SEQ_LEN
import json
import cv2
import mediapipe as mp
import numpy as np

class CFG:
    data_dir = DATA_DIR
    sequence_length = SEQ_LEN
    rows_per_frame = ROWS_PER_FRAME


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
                            mp_drawing.DrawingSpec(color=(227, 224, 113), thickness=3, circle_radius=3),
                            mp_drawing.DrawingSpec(color=(227, 224, 113), thickness=2, circle_radius=2))
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(227, 224, 113), thickness=3, circle_radius=3),
                            mp_drawing.DrawingSpec(color=(227, 224, 113), thickness=2, circle_radius=2))
    
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
