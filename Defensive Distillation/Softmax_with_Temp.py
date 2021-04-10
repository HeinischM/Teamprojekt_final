#My own Softmax layer with Temperature parameter
#This ayer was created modifying the Softmax layer from the Keras package "Layers"
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import math_ops
from tensorflow.python.util.tf_export import keras_export

class Softmax_with_Temp(tf.keras.layers.Layer): 
  """Softmax activation function with temperature.
  Example without mask:
  >>> inp = np.asarray([1., 2., 1.])
  >>> layer = tf.keras.layers.Softmax()
  >>> layer(inp).numpy()
  array([0.21194157, 0.5761169 , 0.21194157], dtype=float32)
  >>> mask = np.asarray([True, False, True], dtype=bool)
  >>> layer(inp, mask).numpy()
  array([0.5, 0. , 0.5], dtype=float32)
  Input shape:
    Arbitrary. Use the keyword argument `input_shape`
    (tuple of integers, does not include the samples axis)
    when using this layer as the first layer in a model.
  Output shape:
    Same shape as the input.
  Arguments:
    axis: Integer, or list of Integers, axis along which the softmax
      normalization is applied.
  Call arguments:
    inputs: The inputs, or logits to the softmax layer.
    mask: A boolean mask of the same shape as `inputs`. Defaults to `None`.
  Returns:
    softmaxed output with the same shape as `inputs`.
  """

  def __init__(self, axis=-1, temperature=1, **kwargs):
    super(Softmax_with_Temp, self).__init__(**kwargs)
    self.supports_masking = True
    self.axis = axis

  def call(self, inputs, mask=None):
    if mask is not None:
      # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
      # masked positions, this operation will create a tensor which is 0.0 for
      # positions we want to attend and -1e.9 for masked positions.
      adder = (1.0 - math_ops.cast(mask, inputs.dtype)) * (
          _large_compatible_negative(inputs.dtype))

      # Since we are adding it to the raw scores before the softmax, this is
      # effectively the same as removing these entirely.
      inputs += adder
    if isinstance(self.axis, (tuple, list)):
      if len(self.axis) > 1:
        #the logits are divided by the parameter Temperature
        return math_ops.exp(inputs/temperature - math_ops.reduce_logsumexp(
            inputs/temperature, axis=self.axis, keepdims=True))
      else:
        return K.softmax(inputs, axis=self.axis[0])
    return K.softmax(inputs, axis=self.axis)

  def get_config(self):
    config = {'axis': self.axis}
    base_config = super(Softmax_with_Temp, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  @tf_utils.shape_type_conversion
  def compute_output_shape(self, input_shape):
    return input_shape



