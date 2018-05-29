r"""The file contains utility methods to compute the mean prediction
 as well as uncertainty metrics given the Monte Carlo samples as input.

 The current implementation contains methods to compute the variation ratio,
 predictive entropy and mutual information uncertainty metrics.

 Useful reference:
 Y. Gal. Uncertainty in Deep Learning. PhD thesis, University of Cambridge, 2016.
"""

import tensorflow as tf

VARIATION_RATIO = 'variation_ratio'
PREDICTIVE_ENTROPY = 'predictive_entropy'
MUTUAL_INFORMATION = 'mutual_information'


def _get_multidimensional_list(batch, height, width):
  """ Creates a list of dimensions [batch, height, width, 1].
  """
  ls = [[[[None]*1]*width]*height]*batch
  return ls


def _create_uni_dim_tensor(input, height_ind, width_ind):
  """ This method expects an input which is a tensor of shape [height, width, channels].
  The method creates a uni-dimensional tensor which contains the values of all the channels
  for a given (height, width) index pair.

  Args:
    input: Tensor of shape [height, width, channels]
    height_ind: index of the height dimension for construction of uni-dimensional tensor
    width_ind: index of the width dimension for construction of uni-dimensional tensor

  Returns:
    A uni-dimensional tensor containing the values collected from all channels in the given
    (height_ind, width_ind) index pair of the input tensor.
  """
  res = []
  num_channels = input.get_shape().as_list()[2]
  for i in range(0, num_channels):
    res.append(input[height_ind][width_ind][i])
  return tf.convert_to_tensor(res)


def _compute_variation_ratio(input):
  """ Utility method to compute variation ratio given an input tensor of shape
  [batch, height, width, num_MC_trials]. It produces an output tensor of the form
  [batch, height, width, 1] where for each image in the batch, the tensor contains
  pixel-wise variation ratio values.
  
  Args:
    input: Tensor of shape [batch, height, width, num_MC_trials]

  Returns:
    Tensor of shape [batch, height, width, 1] containing pixel-wise variation-ratio
    values.
  """
  sh = input.get_shape().as_list()
  batch = sh[0]
  height = sh[1]
  width = sh[2]
  num_mc_trials = sh[3]
  res_arr = _get_multidimensional_list(batch, height, width)
  for i in range(0, batch):
    # reduced_tensor has dimension [height, width, num_MC_trials]
    reduced_tensor = input[i]
    for x in range(0, height):
      for y in range(0, width):
        uni_dim_tensor = _create_uni_dim_tensor(reduced_tensor, x, y)
        y_val, y_indx, y_count = tf.unique_with_counts(uni_dim_tensor)

        # Implementing the variation ratio formula below (1 - fx/T)
        res_arr[i][x][y][0] = 1.0 - (tf.cast(tf.reduce_max(y_count), dtype=tf.float32) / num_mc_trials)
  return tf.convert_to_tensor(res_arr)


def mean_prediction(monte_carlo_samples):
  """ Produces the mean of the given Monte Carlo samples.

  Args:
    monte_carlo_samples: A list of predicted tensors where each element in the list
                         has dimensions [batch, height, width, num_classes].

  Returns:
    The mean of the predictions. The dimension of this tensor is [batch, height, width, num_classes].
    In order to get the actual predictions of the classes, an argmax operation is required on the
    channels.
  """
  return tf.reduce_mean(monte_carlo_samples, 0)


def variation_ratio(monte_carlo_samples):
  """ Produces the pixel-wise variation ratio of the given Monte Carlo samples as an uncertainty metric.

  Args:
    monte_carlo_samples: A list of predicted tensors where each element in the list
                         has dimensions [batch, height, width, num_classes].

  Returns:
    The variation ratio generated from the Monte Carlo samples. The dimension of the output tensor is
    [batch, height, width, 1]. There is just one channel for each image where each pixel contains the
    variation-ratio obtained from the MC samples.
  """
  # Modified MC samples is a list where each element has dimensions [batch, height, width, 1]. Each element
  # in the single channel actually represents the class to which the pixel has been classified.
  modified_mc_samples = []
  for sample in monte_carlo_samples:
    modified_mc_samples.append(tf.expand_dims(tf.argmax(sample, 3), 3))

  # Following tensor has dimensions [batch, height, width, num_mc_trials]
  mod_tensor = tf.concat(modified_mc_samples, axis=3)

  return _compute_variation_ratio(mod_tensor)


def predictive_entropy(monte_carlo_samples):
  """ Produces the pixel-wise predictive entropy of the given Monte Carlo samples as an uncertainty metric.

  Args:
    monte_carlo_samples: A list of predicted tensors where each element in the list
                         has dimensions [batch, height, width, num_classes].

  Returns:
    The predictive entropy generated from the Monte Carlo samples. The dimension of the output tensor is
    [batch, height, width, 1]. There is just one channel for each image where each pixel contains the
    predictive entropy obtained from the MC samples.
  """
  mean_from_trials = tf.reduce_mean(monte_carlo_samples, 0)
  log_mean_from_trials = tf.log(mean_from_trials)
  product = tf.multiply(mean_from_trials, log_mean_from_trials)
  res = -tf.expand_dims(tf.reduce_sum(product, axis=3), 3)
  return res


def mutual_information(monte_carlo_samples):
  """ Produces the pixel-wise mutual information of the given Monte Carlo samples as an uncertainty metric.

  Args:
    monte_carlo_samples: A list of predicted tensors where each element in the list
                           has dimensions [batch, height, width, num_classes].

  Returns:
    The mutual information generated from the Monte Carlo samples. The dimension of the output tensor is
    [batch, height, width, 1]. There is just one channel for each image where each pixel contains the
    predictive entropy obtained from the MC samples.
  """
  processed_monte_carlo_samples = []
  for sample in monte_carlo_samples:
    processed_monte_carlo_samples.append(tf.multiply(sample, tf.log(sample)))
  mean_processed_trials = tf.reduce_mean(processed_monte_carlo_samples, 0)
  sum_part = tf.expand_dims(tf.reduce_sum(mean_processed_trials, axis=3), 3)
  return predictive_entropy(monte_carlo_samples) + sum_part