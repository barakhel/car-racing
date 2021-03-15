import gin
import tensorflow as tf
from tf_agents.keras_layers import dynamic_unroll_layer
from tf_agents.networks import network
from tf_agents.networks import utils
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.utils import nest_utils

from tensorflow.python.util import nest

def _copy_layer(layer):
  if not isinstance(layer, tf.keras.layers.Layer):
    raise TypeError('layer is not a keras layer: %s' % str(layer))

  # pylint:disable=unidiomatic-typecheck
  if type(layer) == tf.compat.v1.keras.layers.DenseFeatures:
    raise ValueError('DenseFeatures V1 is not supported. '
                     'Use tf.compat.v2.keras.layers.DenseFeatures instead.')
    
  return type(layer).from_config(layer.get_config())


@gin.configurable
class CriticRnnNetwork(network.Network):

  def __init__(self,
               input_tensor_spec,
               preprocessing_layer=None,
               action_fc_layer_params=(200,),
               joint_fc_layer_params=(100,),
               lstm_size=(40,),
               output_fc_layer_params=(200, 100),
               activation_fn=tf.keras.activations.relu,
               name='CriticRnnNetwork'):
    
    observation_spec, action_spec = input_tensor_spec

    if len(tf.nest.flatten(action_spec)) > 1:
      raise ValueError('Only a single action is supported by this network.')

    if preprocessing_layer:
        preprocessing_layer = _copy_layer(preprocessing_layer)


    action_layers = utils.mlp_layers(
        None,
        action_fc_layer_params,
        activation_fn=activation_fn,
        kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(
            scale=1. / 3., mode='fan_in', distribution='uniform'),
        name='action_encoding')

    joint_layers = utils.mlp_layers(
        None,
        joint_fc_layer_params,
        activation_fn=activation_fn,
        kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(
            scale=1. / 3., mode='fan_in', distribution='uniform'),
        name='joint_mlp')

    # Create RNN cell
    if len(lstm_size) == 1:
      cell = tf.keras.layers.LSTMCell(lstm_size[0])
    else:
      cell = tf.keras.layers.StackedRNNCells(
          [tf.keras.layers.LSTMCell(size) for size in lstm_size])

    counter = [-1]

    def create_spec(size):
      counter[0] += 1
      return tensor_spec.TensorSpec(
          size, dtype=tf.float32, name='network_state_%d' % counter[0])

    state_spec = tf.nest.map_structure(create_spec, cell.state_size)

    output_layers = utils.mlp_layers(fc_layer_params=output_fc_layer_params,
                                     name='output')

    output_layers.append(
        tf.keras.layers.Dense(
            1,
            activation=None,
            kernel_initializer=tf.keras.initializers.RandomUniform(
                minval=-0.003, maxval=0.003),
            name='value'))

    super(CriticRnnNetwork, self).__init__(
        input_tensor_spec=input_tensor_spec,
        state_spec=state_spec,
        name=name)

    self._action_layers = action_layers
    self._joint_layers = joint_layers
    self._dynamic_unroll = dynamic_unroll_layer.DynamicUnroll(cell)
    self._output_layers = output_layers

    self._preprocessing_layer = preprocessing_layer

  def call(self, inputs, step_type, network_state=None, training=False):
    observation, action = inputs
    observation_spec, _ = self.input_tensor_spec

    
    if self._preprocessing_layer:
      observation = self._preprocessing_layer(observation,training=training)

    observation_spec = tensor_spec.TensorSpec((observation.shape[-1],), dtype=observation.dtype)

    num_outer_dims = nest_utils.get_outer_rank(observation,
                                               observation_spec)
    if num_outer_dims not in (1, 2):
      raise ValueError(
          'Input observation must have a batch or batch x time outer shape.')

    has_time_dim = num_outer_dims == 2
    if not has_time_dim:
      # Add a time dimension to the inputs.
      observation = tf.expand_dims(observation, 1)
      action = tf.expand_dims(action, 1)
      step_type = tf.expand_dims(step_type, 1)

    observation = tf.cast(observation, tf.float32)
    action = tf.cast(action, tf.float32)

    batch_squash = utils.BatchSquash(2)  # Squash B, and T dims.
    observation = batch_squash.flatten(observation)  # [B, T, ...] -> [BxT, ...]
    action = batch_squash.flatten(action)

    for layer in self._action_layers:
      action = layer(action, training=training)

    joint = tf.concat([observation, action], -1)
    for layer in self._joint_layers:
      joint = layer(joint, training=training)

    joint = batch_squash.unflatten(joint)  # [B x T, ...] -> [B, T, ...]

    with tf.name_scope('reset_mask'):
      reset_mask = tf.equal(step_type, ts.StepType.FIRST)
    # Unroll over the time sequence.
    joint, network_state = self._dynamic_unroll(
        joint,
        initial_state=network_state,
        reset_mask=reset_mask,
        training=training)

    output = batch_squash.flatten(joint)  # [B, T, ...] -> [B x T, ...]

    for layer in self._output_layers:
      output = layer(output, training=training)

    q_value = tf.reshape(output, [-1])
    q_value = batch_squash.unflatten(q_value)  # [B x T, ...] -> [B, T, ...]
    if not has_time_dim:
      q_value = tf.squeeze(q_value, axis=1)

    return q_value, network_state