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
  """ Produces the mean of the given Monte Carlo samples as the

  Args:
    monte_carlo_samples: A list of predicted tensors where each element in the list
                         has dimensions [batch, height, width, num_classes].

  Returns:
    The mean of the predictions. The dimension of this tensor is [batch, height, width, num_classes].
  """
  return tf.reduce_mean(monte_carlo_samples, 0)


