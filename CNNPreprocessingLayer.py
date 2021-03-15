import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D


class CNNPreprocessingLayer(tf.keras.layers.Layer):
  """Preprocessing layers for multiple source inputs."""

  def __init__(self, base_depth=8, name=None):
    super(CNNPreprocessingLayer, self).__init__(name=name)
    self.base_depth = base_depth
    self.conv1 = Conv2D(base_depth,7,3,padding="SAME", activation=tf.nn.relu)
    self.polling1 = MaxPool2D(padding="SAME")
    self.conv2 = Conv2D(2 * base_depth,4,3,padding="VALID", activation=tf.nn.relu)
    self.polling2 = MaxPool2D(padding="SAME")

  def __call__(self, image, training=None):
    image = tf.image.convert_image_dtype(image, tf.float32)
    image_shape = tf.shape(image)[-3:]
#     print('shsh',tf.shape(image), image.get_shape())
    collapsed_shape = tf.concat(([-1], image_shape), axis=0)
    out = tf.reshape(image, collapsed_shape)  # (sample*N*T, h, w, c)
    out = tf.image.crop_to_bounding_box(out,0,6,84,84)
    out = tf.image.rgb_to_grayscale(out)
    out = out * 2. - 1.
    out = self.conv1(out)
    out = self.polling1(out)
    out = self.conv2(out)
    out = self.polling2(out)
    expanded_shape = tf.concat((tf.shape(image)[:-3], [-1]), axis=0)
    return tf.reshape(out, expanded_shape)  # (sample, N, T, feature)

  def get_config(self):
    config = {
      'base_depth':self.base_depth}
    return config

