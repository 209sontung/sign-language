from .utils import Preprocess
from .config import MAX_LEN, CHANNELS, NUM_CLASSES
import tensorflow as tf


class ECA(tf.keras.layers.Layer):
    """
    Efficient Channel Attention layer.

    Args:
        kernel_size (int): Size of the kernel for the convolutional layer.

    Returns:
        Output tensor after applying the efficient channel attention mechanism.
    """

    def __init__(self, kernel_size=5, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.kernel_size = kernel_size
        self.conv = tf.keras.layers.Conv1D(1, kernel_size=kernel_size, strides=1, padding="same", use_bias=False)

    def call(self, inputs, mask=None):
        """
        Applies the efficient channel attention mechanism to the input tensor.

        Args:
            inputs: Input tensor.
            mask: Mask tensor for masking specific values in the input.

        Returns:
            Output tensor after applying the efficient channel attention mechanism.
        """
        nn = tf.keras.layers.GlobalAveragePooling1D()(inputs, mask=mask)
        nn = tf.expand_dims(nn, -1)
        nn = self.conv(nn)
        nn = tf.squeeze(nn, -1)
        nn = tf.nn.sigmoid(nn)
        nn = nn[:,None,:]
        return inputs * nn

class LateDropout(tf.keras.layers.Layer):
    """
    Layer that applies dropout after a certain training step.

    Args:
        rate (float): Dropout rate.
        noise_shape: Shape of the binary dropout mask.
        start_step (int): The training step after which the dropout is applied.

    Returns:
        Output tensor after applying dropout.
    """
    def __init__(self, rate, noise_shape=None, start_step=0, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.rate = rate
        self.start_step = start_step
        self.dropout = tf.keras.layers.Dropout(rate, noise_shape=noise_shape)

    def build(self, input_shape):
        super().build(input_shape)
        agg = tf.VariableAggregation.ONLY_FIRST_REPLICA
        self._train_counter = tf.Variable(0, dtype="int64", aggregation=agg, trainable=False)

    def call(self, inputs, training=False):
        """
        Applies dropout to the input tensor.

        Args:
            inputs: Input tensor.
            training (bool): Whether the model is in training mode or not.

        Returns:
            Output tensor after applying dropout.
        """
        x = tf.cond(self._train_counter < self.start_step, lambda:inputs, lambda:self.dropout(inputs, training=training))
        if training:
            self._train_counter.assign_add(1)
        return x

class CausalDWConv1D(tf.keras.layers.Layer):
    """
    Causal Dilated Depthwise Convolutional 1D layer.

    Args:
        kernel_size (int): Size of the kernel for the convolutional layer.
        dilation_rate (int): Dilation rate for the convolutional layer.
        use_bias (bool): Whether to use bias in the convolutional layer.
        depthwise_initializer: Initializer for the depthwise convolutional kernel.
        name (str): Name of the layer.

    Returns:
        Output tensor after applying the causal dilated depthwise convolution.
    """
    
    def __init__(self, 
        kernel_size=17,
        dilation_rate=1,
        use_bias=False,
        depthwise_initializer='glorot_uniform',
        name='', **kwargs):
        super().__init__(name=name,**kwargs)
        self.causal_pad = tf.keras.layers.ZeroPadding1D((dilation_rate*(kernel_size-1),0),name=name + '_pad')
        self.dw_conv = tf.keras.layers.DepthwiseConv1D(
                            kernel_size,
                            strides=1,
                            dilation_rate=dilation_rate,
                            padding='valid',
                            use_bias=use_bias,
                            depthwise_initializer=depthwise_initializer,
                            name=name + '_dwconv')
        self.supports_masking = True
        
    def call(self, inputs):
        """
        Applies the causal dilated depthwise convolution to the input tensor.

        Args:
            inputs: Input tensor.

        Returns:
            Output tensor after applying the causal dilated depthwise convolution.
        """
        x = self.causal_pad(inputs)
        x = self.dw_conv(x)
        return x

def Conv1DBlock(channel_size,
                kernel_size,
                dilation_rate=1,
                drop_rate=0.0,
                expand_ratio=2,
                se_ratio=0.25,
                activation='swish',
name=None):
    """
    Efficient Conv1D block, @hoyso48
    
    Args:
        channel_size (int): Number of output channels for the block.
        kernel_size (int): Size of the kernel for the convolutional layers.
        dilation_rate (int): Dilation rate for the convolutional layers.
        drop_rate (float): Dropout rate.
        expand_ratio (int): Expansion ratio for the Dense layer.
        se_ratio (float): Squeeze-and-Excitation ratio.
        activation (str): Activation function.
        name (str): Name of the block.

    Returns:
        Function to apply the Conv1D block to an input tensor.
    """

    if name is None:
        name = str(tf.keras.backend.get_uid("mbblock"))
    # Expansion phase
    def apply(inputs):
        channels_in = tf.keras.backend.int_shape(inputs)[-1]
        channels_expand = channels_in * expand_ratio

        skip = inputs

        x = tf.keras.layers.Dense(
            channels_expand,
            use_bias=True,
            activation=activation,
            name=name + '_expand_conv')(inputs)

        # Depthwise Convolution
        x = CausalDWConv1D(kernel_size,
            dilation_rate=dilation_rate,
            use_bias=False,
            name=name + '_dwconv')(x)

        x = tf.keras.layers.BatchNormalization(momentum=0.95, name=name + '_bn')(x)

        x  = ECA()(x)

        x = tf.keras.layers.Dense(
            channel_size,
            use_bias=True,
            name=name + '_project_conv')(x)

        if drop_rate > 0:
            x = tf.keras.layers.Dropout(drop_rate, noise_shape=(None,1,1), name=name + '_drop')(x)

        if (channels_in == channel_size):
            x = tf.keras.layers.add([x, skip], name=name + '_add')
        return x

    return apply

class MultiHeadSelfAttention(tf.keras.layers.Layer):
    """
    Multi-Head Self-Attention layer.
    
    Args:
        dim (int): Dimension of the attention vectors.
        num_heads (int): Number of attention heads.
        dropout (float): Dropout rate.

    Returns:
        Output tensor after applying the multi-head self-attention mechanism.
    """
    def __init__(self, dim=256, num_heads=4, dropout=0, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.scale = self.dim ** -0.5
        self.num_heads = num_heads
        self.qkv = tf.keras.layers.Dense(3 * dim, use_bias=False)
        self.drop1 = tf.keras.layers.Dropout(dropout)
        self.proj = tf.keras.layers.Dense(dim, use_bias=False)
        self.supports_masking = True

    def call(self, inputs, mask=None):
        """
        Applies the multi-head self-attention mechanism to the input tensor.

        Args:
            inputs: Input tensor.
            mask: Mask tensor indicating valid positions.

        Returns:
            Output tensor after applying the multi-head self-attention mechanism.
        """
        qkv = self.qkv(inputs)
        qkv = tf.keras.layers.Permute((2, 1, 3))(tf.keras.layers.Reshape((-1, self.num_heads, self.dim * 3 // self.num_heads))(qkv))
        q, k, v = tf.split(qkv, [self.dim // self.num_heads] * 3, axis=-1)

        attn = tf.matmul(q, k, transpose_b=True) * self.scale

        if mask is not None:
            mask = mask[:, None, None, :]

        attn = tf.keras.layers.Softmax(axis=-1)(attn, mask=mask)
        attn = self.drop1(attn)

        x = attn @ v
        x = tf.keras.layers.Reshape((-1, self.dim))(tf.keras.layers.Permute((2, 1, 3))(x))
        x = self.proj(x)
        return x


def TransformerBlock(dim=256, num_heads=4, expand=4, attn_dropout=0.2, drop_rate=0.2, activation='swish'):
    """
    Transformer Block.
    
    Args:
        dim (int): Dimension of the attention vectors.
        num_heads (int): Number of attention heads.
        expand (int): Expansion ratio for the Dense layer.
        attn_dropout (float): Dropout rate for attention mechanism.
        drop_rate (float): Dropout rate.
        activation (str): Activation function.

    Returns:
        Function to apply the Transformer Block to an input tensor.
    """
    def apply(inputs):
        x = inputs
        x = tf.keras.layers.BatchNormalization(momentum=0.95)(x)
        x = MultiHeadSelfAttention(dim=dim,num_heads=num_heads,dropout=attn_dropout)(x)
        x = tf.keras.layers.Dropout(drop_rate, noise_shape=(None,1,1))(x)
        x = tf.keras.layers.Add()([inputs, x])
        attn_out = x

        x = tf.keras.layers.BatchNormalization(momentum=0.95)(x)
        x = tf.keras.layers.Dense(dim*expand, use_bias=False, activation=activation)(x)
        x = tf.keras.layers.Dense(dim, use_bias=False)(x)
        x = tf.keras.layers.Dropout(drop_rate, noise_shape=(None,1,1))(x)
        x = tf.keras.layers.Add()([attn_out, x])
        return x
    return apply

class TFLiteModel(tf.Module):
    """
    TensorFlow Lite model that takes input tensors and applies:
        – A Preprocessing Model
        – The ISLR model 
    """

    def __init__(self, islr_models):
        """
        Initializes the TFLiteModel with the specified preprocessing model and ISLR model.
        """
        super(TFLiteModel, self).__init__()

        # Load the feature generation and main models
        self.prep_inputs = Preprocess()
        self.islr_models   = islr_models
    
    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 543, 3], dtype=tf.float32, name='inputs')])
    def __call__(self, inputs):
        """
        Applies the feature generation model and main model to the input tensors.

        Args:
            inputs: Input tensor with shape [batch_size, 543, 3].

        Returns:
            A dictionary with a single key 'outputs' and corresponding output tensor.
        """
        x = self.prep_inputs(tf.cast(inputs, dtype=tf.float32))
        outputs = [model(x) for model in self.islr_models]
        outputs = tf.keras.layers.Average()(outputs)[0]
        return {'outputs': outputs}

def get_model(max_len=MAX_LEN, dropout_step=0, dim=192):
    """
    Creates a model for sequence classification using a combination of convolutional layers and transformer blocks.

    Args:
        max_len (int): Maximum length of the input sequence.
        dropout_step (int): Dropout step for the LateDropout layer.
        dim (int): Dimension of the hidden representations.

    Returns:
        A TensorFlow Keras Model object.
    """
    inp = tf.keras.Input((max_len,CHANNELS))
    #x = tf.keras.layers.Masking(mask_value=PAD,input_shape=(max_len,CHANNELS))(inp) #we don't need masking layer with inference
    x = inp
    ksize = 17
    
    # Stem layers
    x = tf.keras.layers.Dense(dim, use_bias=False,name='stem_conv')(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.95,name='stem_bn')(x)

    # Convolutional and Transformer blocks
    x = Conv1DBlock(dim,ksize,drop_rate=0.2)(x)
    x = Conv1DBlock(dim,ksize,drop_rate=0.2)(x)
    x = Conv1DBlock(dim,ksize,drop_rate=0.2)(x)
    x = TransformerBlock(dim,expand=2)(x)

    x = Conv1DBlock(dim,ksize,drop_rate=0.2)(x)
    x = Conv1DBlock(dim,ksize,drop_rate=0.2)(x)
    x = Conv1DBlock(dim,ksize,drop_rate=0.2)(x)
    x = TransformerBlock(dim,expand=2)(x)

    # Additional convolutional blocks and transformer blocks for larger models
    if dim == 384: #for the 4x sized model
        x = Conv1DBlock(dim,ksize,drop_rate=0.2)(x)
        x = Conv1DBlock(dim,ksize,drop_rate=0.2)(x)
        x = Conv1DBlock(dim,ksize,drop_rate=0.2)(x)
        x = TransformerBlock(dim,expand=2)(x)

        x = Conv1DBlock(dim,ksize,drop_rate=0.2)(x)
        x = Conv1DBlock(dim,ksize,drop_rate=0.2)(x)
        x = Conv1DBlock(dim,ksize,drop_rate=0.2)(x)
        x = TransformerBlock(dim,expand=2)(x)

    # Top layers
    x = tf.keras.layers.Dense(dim*2,activation=None,name='top_conv')(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = LateDropout(0.8, start_step=dropout_step)(x)
    x = tf.keras.layers.Dense(NUM_CLASSES,name='classifier', activation = 'softmax')(x)
    return tf.keras.Model(inp, x)
