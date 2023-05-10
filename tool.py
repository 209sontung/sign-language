# from backbone import Conv1DBlock, TransformerBlock, LateDropout
# from config import MAX_LEN, CHANNELS, NUM_CLASSES, POINT_LANDMARKS

# import tensorflow as tf
# import json
# import pandas as pd
# import numpy as np


# ROWS_PER_FRAME = 543  # number of landmarks per frame

# def load_relevant_data_subset(pq_path):
#     data_columns = ['x', 'y', 'z']
#     data = pd.read_parquet(pq_path, columns=data_columns)
#     n_frames = int(len(data) / ROWS_PER_FRAME)
#     data = data.values.reshape(n_frames, ROWS_PER_FRAME, len(data_columns))
#     return data.astype(np.float32)

# def tf_nan_mean(x, axis=0, keepdims=False):
#     return tf.reduce_sum(tf.where(tf.math.is_nan(x), tf.zeros_like(x), x), axis=axis, keepdims=keepdims) / tf.reduce_sum(tf.where(tf.math.is_nan(x), tf.zeros_like(x), tf.ones_like(x)), axis=axis, keepdims=keepdims)

# def tf_nan_std(x, center=None, axis=0, keepdims=False):
#     if center is None:
#         center = tf_nan_mean(x, axis=axis,  keepdims=True)
#     d = x - center
#     return tf.math.sqrt(tf_nan_mean(d * d, axis=axis, keepdims=keepdims))

# class Preprocess(tf.keras.layers.Layer):
#     def __init__(self, max_len=MAX_LEN, point_landmarks=POINT_LANDMARKS, **kwargs):
#         super().__init__(**kwargs)
#         self.max_len = max_len
#         self.point_landmarks = point_landmarks

#     def call(self, inputs):
#         if tf.rank(inputs) == 3:
#             x = inputs[None,...]
#         else:
#             x = inputs
        
#         mean = tf_nan_mean(tf.gather(x, [17], axis=2), axis=[1,2], keepdims=True)
#         mean = tf.where(tf.math.is_nan(mean), tf.constant(0.5,x.dtype), mean)
#         x = tf.gather(x, self.point_landmarks, axis=2) #N,T,P,C
#         std = tf_nan_std(x, center=mean, axis=[1,2], keepdims=True)
        
#         x = (x - mean)/std

#         if self.max_len is not None:
#             x = x[:,:self.max_len]
#         length = tf.shape(x)[1]
#         x = x[...,:2]

#         dx = tf.cond(tf.shape(x)[1]>1,lambda:tf.pad(x[:,1:] - x[:,:-1], [[0,0],[0,1],[0,0],[0,0]]),lambda:tf.zeros_like(x))

#         dx2 = tf.cond(tf.shape(x)[1]>2,lambda:tf.pad(x[:,2:] - x[:,:-2], [[0,0],[0,2],[0,0],[0,0]]),lambda:tf.zeros_like(x))

#         x = tf.concat([
#             tf.reshape(x, (-1,length,2*len(self.point_landmarks))),
#             tf.reshape(dx, (-1,length,2*len(self.point_landmarks))),
#             tf.reshape(dx2, (-1,length,2*len(self.point_landmarks))),
#         ], axis = -1)
        
#         x = tf.where(tf.math.is_nan(x),tf.constant(0.,x.dtype),x)
        
#         return x
    
# def get_model(max_len=MAX_LEN, dropout_step=0, dim=192):
#     inp = tf.keras.Input((max_len,CHANNELS))
#     #x = tf.keras.layers.Masking(mask_value=PAD,input_shape=(max_len,CHANNELS))(inp) #we don't need masking layer with inference
#     x = inp
#     ksize = 17
#     x = tf.keras.layers.Dense(dim, use_bias=False,name='stem_conv')(x)
#     x = tf.keras.layers.BatchNormalization(momentum=0.95,name='stem_bn')(x)

#     x = Conv1DBlock(dim,ksize,drop_rate=0.2)(x)
#     x = Conv1DBlock(dim,ksize,drop_rate=0.2)(x)
#     x = Conv1DBlock(dim,ksize,drop_rate=0.2)(x)
#     x = TransformerBlock(dim,expand=2)(x)

#     x = Conv1DBlock(dim,ksize,drop_rate=0.2)(x)
#     x = Conv1DBlock(dim,ksize,drop_rate=0.2)(x)
#     x = Conv1DBlock(dim,ksize,drop_rate=0.2)(x)
#     x = TransformerBlock(dim,expand=2)(x)

#     if dim == 384: #for the 4x sized model
#         x = Conv1DBlock(dim,ksize,drop_rate=0.2)(x)
#         x = Conv1DBlock(dim,ksize,drop_rate=0.2)(x)
#         x = Conv1DBlock(dim,ksize,drop_rate=0.2)(x)
#         x = TransformerBlock(dim,expand=2)(x)

#         x = Conv1DBlock(dim,ksize,drop_rate=0.2)(x)
#         x = Conv1DBlock(dim,ksize,drop_rate=0.2)(x)
#         x = Conv1DBlock(dim,ksize,drop_rate=0.2)(x)
#         x = TransformerBlock(dim,expand=2)(x)

#     x = tf.keras.layers.Dense(dim*2,activation=None,name='top_conv')(x)
#     x = tf.keras.layers.GlobalAveragePooling1D()(x)
#     x = LateDropout(0.8, start_step=dropout_step)(x)
#     x = tf.keras.layers.Dense(NUM_CLASSES,name='classifier')(x)
#     return tf.keras.Model(inp, x)

# def read_json_file(file_path):
#     """Read a JSON file and parse it into a Python object.

#     Args:
#         file_path (str): The path to the JSON file to read.

#     Returns:
#         dict: A dictionary object representing the JSON data.
        
#     Raises:
#         FileNotFoundError: If the specified file path does not exist.
#         ValueError: If the specified file path does not contain valid JSON data.
#     """
#     try:
#         # Open the file and load the JSON data into a Python object
#         with open(file_path, 'r') as file:
#             json_data = json.load(file)
#         return json_data
#     except FileNotFoundError:
#         # Raise an error if the file path does not exist
#         raise FileNotFoundError(f"File not found: {file_path}")
#     except ValueError:
#         # Raise an error if the file does not contain valid JSON data
#         raise ValueError(f"Invalid JSON data in file: {file_path}")
    