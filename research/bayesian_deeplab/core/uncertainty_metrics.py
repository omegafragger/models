r"""The file contains utility methods to compute the mean prediction
 as well as uncertainty metrics given the Monte Carlo samples as input.

 The current implementation contains methods to compute the variation ratio,
 predictive entropy and mutual information uncertainty metrics.

 Useful reference:
 Y. Gal. Uncertainty in Deep Learning. PhD thesis, University of Cambridge, 2016.
"""

import tensorflow as tf

_VARIATION_RATIO = 'variation_ratio'
_PREDICTIVE_ENTROPY = 'predictive_entropy'
_MUTUAL_INFORMATION = 'mutual_information'


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
  num_mc_trials = len(monte_carlo_samples)

  # Getting the tensor shape
  shape = monte_carlo_samples[0].get_shape().as_list()
  batch = shape[0]

  # Modified MC samples is a list where each element has dimensions [batch, height, width, 1]. Each element
  # in the single channel actually represents the class to which the pixel has been classified.
  modified_mc_samples = []
  for i in range(0, num_mc_trials):
    modified_mc_samples[i] = tf.argmax(monte_carlo_samples[i], 3)

  # Following tensor has dimensions [batch, height, width, num_mc_trials]
  mod_tensor = tf.concat(modified_mc_samples, axis=3)



