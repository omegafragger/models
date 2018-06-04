r"""The file contains utility methods to compute the mean prediction
 as well as uncertainty metrics given the Monte Carlo samples as input.

 The current implementation contains methods to compute the predictive entropy
 and mutual information uncertainty metrics.

 Useful reference:
 Y. Gal. Uncertainty in Deep Learning. PhD thesis, University of Cambridge, 2016.
"""

import tensorflow as tf
import math

PREDICTIVE_ENTROPY = 'predictive_entropy'
MUTUAL_INFORMATION = 'mutual_information'


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
  sh = mean_from_trials.get_shape().as_list()
  jitter = tf.fill(sh, math.exp(-6))
  log_mean_from_trials = tf.log(mean_from_trials + jitter)
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
    sh = sample.get_shape().as_list()
    jitter = tf.fill(sh, math.exp(-6))
    processed_monte_carlo_samples.append(tf.multiply(sample, tf.log(sample + jitter)))
  mean_processed_trials = tf.reduce_mean(processed_monte_carlo_samples, 0)
  sum_part = tf.expand_dims(tf.reduce_sum(mean_processed_trials, axis=3), 3)
  return predictive_entropy(monte_carlo_samples) + sum_part