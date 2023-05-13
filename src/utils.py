from .config import MAX_LEN, POINT_LANDMARKS
import tensorflow as tf

def tf_nan_mean(x, axis=0, keepdims=False):
    """
    Computes the mean of the input tensor while ignoring NaN values.

    Args:
        x: Input tensor.
        axis: Axis along which to compute the mean. Default is 0.
        keepdims: Whether to keep the dimensions of the input tensor. Default is False.

    Returns:
        The mean of the input tensor with NaN values ignored.
    """
    return tf.reduce_sum(tf.where(tf.math.is_nan(x), tf.zeros_like(x), x), axis=axis, keepdims=keepdims) / tf.reduce_sum(tf.where(tf.math.is_nan(x), tf.zeros_like(x), tf.ones_like(x)), axis=axis, keepdims=keepdims)

def tf_nan_std(x, center=None, axis=0, keepdims=False):
    """
    Computes the standard deviation of the input tensor while ignoring NaN values.

    Args:
        x: Input tensor.
        center: Tensor representing the mean of the input tensor. If None, the mean is computed internally.
        axis: Axis along which to compute the standard deviation. Default is 0.
        keepdims: Whether to keep the dimensions of the input tensor. Default is False.

    Returns:
        The standard deviation of the input tensor with NaN values ignored.
    """
    if center is None:
        center = tf_nan_mean(x, axis=axis,  keepdims=True)
    d = x - center
    return tf.math.sqrt(tf_nan_mean(d * d, axis=axis, keepdims=keepdims))

class Preprocess(tf.keras.layers.Layer):
    """
    Preprocessing layer for input data.

    Args:
        max_len: Maximum length of the input sequence. Default is MAX_LEN from config.
        point_landmarks: List of point landmarks to extract from the input. Default is POINT_LANDMARKS from config.
    """
    
    def __init__(self, max_len=MAX_LEN, point_landmarks=POINT_LANDMARKS, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.point_landmarks = point_landmarks

    def call(self, inputs):
        """
        Preprocesses the input data.

        Args:
            inputs: Input tensor.

        Returns:
            Preprocessed tensor.
        """
        if tf.rank(inputs) == 3:
            x = inputs[None,...]
        else:
            x = inputs
        
        mean = tf_nan_mean(tf.gather(x, [17], axis=2), axis=[1,2], keepdims=True)
        mean = tf.where(tf.math.is_nan(mean), tf.constant(0.5,x.dtype), mean)
        x = tf.gather(x, self.point_landmarks, axis=2) #N,T,P,C
        std = tf_nan_std(x, center=mean, axis=[1,2], keepdims=True)
        
        x = (x - mean)/std

        if self.max_len is not None:
            x = x[:,:self.max_len]
        length = tf.shape(x)[1]
        x = x[...,:2]

        dx = tf.cond(tf.shape(x)[1]>1,lambda:tf.pad(x[:,1:] - x[:,:-1], [[0,0],[0,1],[0,0],[0,0]]),lambda:tf.zeros_like(x))

        dx2 = tf.cond(tf.shape(x)[1]>2,lambda:tf.pad(x[:,2:] - x[:,:-2], [[0,0],[0,2],[0,0],[0,0]]),lambda:tf.zeros_like(x))

        x = tf.concat([
            tf.reshape(x, (-1,length,2*len(self.point_landmarks))),
            tf.reshape(dx, (-1,length,2*len(self.point_landmarks))),
            tf.reshape(dx2, (-1,length,2*len(self.point_landmarks))),
        ], axis = -1)
        
        x = tf.where(tf.math.is_nan(x),tf.constant(0.,x.dtype),x)
        
        return x
    
