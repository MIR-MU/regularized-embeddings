from functools import total_ordering
import logging
import lzma
from math import sqrt
import pickle
import subprocess

import numpy as np
import scipy.stats
import sklearn.metrics

LOGGER = logging.getLogger(__name__)


def make(target):
    """Uses GNU make to produce a target.

    Parameters
    ----------
    target : str
        The name of the target.
    """

    command = 'make {}'.format(target)
    return_code = subprocess.call(command, shell=True)
    assert return_code == 0, return_code


def binomial_confidence_interval(num_successes, num_trials, significance_level):
    """Computes a Wald confidence interval for the parameter p of a binomial random variable.
    Given a sample of Bernoulli trials, we approximate an adjusted Wald confidence interval for the
    population success probability :math:`p` of a binomial random variable using the central limit
    theorem. The Wald interval was first described by [Simon12]_ and the adjustment for small
    samples was proposed by [AgrestiCouli98]_.
    .. [Simon12] Laplace, Pierre Simon (1812). Théorie analytique des probabilités (in French). p.
       283.
    .. [AgrestiCouli98] Agresti, Alan; Coull, Brent A. (1998). "Approximate is better than 'exact'
       for interval estimation of binomial proportions". The American Statistician. 52: 119–126.
       doi:10.2307/2685469.
    Parameters
    ----------
    num_successes : int
        The number of successful Bernoulli trials in the sample.
    num_trials : int
        The sample size.
    significance_level : scalar
        The likelihood that an observation of the random variable falls into the confidence
        interval.
    Returns
    -------
    pointwise_estimate : scalar
        An unbiased pointwise estimate of the expected value of the binomial random variable.
    lower_bound : scalar
        The lower bound of the confidence interval.
    upper_bound : scalar
        The upper bound of the confidence interval.
    Raises
    ------
    ValueError
        If the number of trials is less than or equal to zero, or the number of successes is greater
        than the number of trials.
    """

    if num_trials <= 0:
        raise ValueError('The number of trials is less than or equal to zero')
    if num_successes > num_trials:
        raise ValueError('The number of successes is greater than the number of trials')

    z = scipy.stats.norm.ppf(1 - significance_level / 2)
    z2 = z**2
    n = num_trials + z2
    p = (num_successes + z2 / 2) / n
    radius = z * sqrt(p * (1 - p) / n)
    lower_bound, upper_bound = np.clip((p - radius, p + radius), 0, 1)
    return (num_successes / num_trials, lower_bound, upper_bound)


@total_ordering
class ClassificationResult(object):
    """A classification result.

    Parameters
    ----------
    confusion_matrix : array_like
        A confusion matrix.
    params : dict
        A dict of params related to the classification result.

    Attributes
    ----------
    confusion_matrix : array_like
        A confusion matrix.
    params : dict
        A dict of params related to the classification result.
    """
    def __init__(self, confusion_matrix, params, **kwargs):
        self.confusion_matrix = confusion_matrix
        self.params = params
        num_successes = np.diag(confusion_matrix).sum()
        num_trials = np.sum(confusion_matrix)
        accuracy = num_successes / num_trials
        self._accuracy = accuracy

    @staticmethod
    def from_results(y_true, y_pred, params):
        """Produces a classification result from predictions and ground truth.

        Parameters
        ----------
        y_true : array_like
            Ground truth (correct) labels.
        y_pred : array_like
            Predicted labels, as returned by a classifier.
        params : dict
            A dict of params related to the classification result.

        Returns
        -------
        result : ClassificationResult
            The classification result produced from predictions and ground truth.
        """

        confusion_matrix = sklearn.metrics.confusion_matrix(y_true, y_pred)
        params = dict(params)
        params.pop('collection_corpus', None)
        params.pop('query_corpus', None)
        result = ClassificationResult(confusion_matrix, params)
        return result

    @staticmethod
    def from_file(basename):
        """Loads a classification result from a file.

        Parameters
        ----------
        basename : str
            The base of the filename.

        Returns
        -------
        result : ClassificationResult
            The classification result loaded from the file.
        """
        filename = 'results/classification-{}.pkl.xz'.format(basename)
        with lzma.open(filename, 'rb') as f:
            LOGGER.info('Loading a classification result from file {}.'.format(filename))
            result = ClassificationResult(**pickle.load(f))
        return result

    def to_file(self, basename):
        """Saves a classification result to a file.

        Parameters
        ----------
        basename : str
            The base of the filename.
        """
        make('results')
        filename = 'results/classification-{}.pkl.xz'.format(basename)
        with lzma.open(filename, 'wb', preset=0) as f:
            LOGGER.info('Saving classification result to file {}.'.format(filename))
            pickle.dump(vars(self), f, 4)

    @staticmethod
    def from_similarities(similarities, source_dataset, target_dataset, params):
        """Produces a classification result from similarities between datasets.

        Parameters
        ----------
        similarities : np.matrix
            A matrix of similarities between two datasets.
        source_dataset : Dataset
            The source dataset.
        target_dataset : Dataset
            The target dataset.
        params : dict
            A dict of params related to the classification result.

        Returns
        -------
        result : ClassificationResult
            The classification result.
        """

        k = params['k']
        topk_documents = np.argpartition(similarities, -k)[:, -k:]
        topk_targets = np.take(source_dataset.target, topk_documents)
        y_true = target_dataset.target
        y_pred = scipy.stats.mode(topk_targets, axis=1)[0].T[0]
        result = ClassificationResult.from_results(y_true, y_pred, params)
        return result

    def accuracy(self, significance_level=0.05):
        """Returns pointwise and interval estimates for the accuracy.

        We assume that the trials are binomial.

        Parameters
        ----------
        significance_level : scalar
            The likelihood that the actual accuracy falls into the
            confidence interval.

        Returns
        -------
        pointwise_estimate : scalar
            An unbiased pointwise estimate of the expected value of
            the accuracy.
        lower_bound : scalar
            The lower bound of the confidence interval for the accuracy.
        upper_bound : scalar
            The upper bound of the confidence interval for the accuracy.
        """
        num_successes = np.diag(self.confusion_matrix).sum()
        num_trials = np.sum(self.confusion_matrix)
        return binomial_confidence_interval(num_successes, num_trials, significance_level)

    def __eq__(self, other):
        if isinstance(other, ClassificationResult):
            return self._accuracy == other._accuracy
        return NotImplemented

    def __lt__(self, other):
        if isinstance(other, ClassificationResult):
            return self._accuracy < other._accuracy
        return NotImplemented

    def __repr__(self):
        return '<ClassificationResult, accuracy: {:.02f}%, params: {}>'.format(self._accuracy * 100, self.params)


class KusnerEtAlClassificationResult(object):
    """A classification result taken from the Kusner et al. (2015) paper.

    Parameters
    ----------
    test_error_height : scalar
        The height in pixels of a reported test error in Figure 3 of Kusner et al. (2015).
    error_bar_height : scalar
        The height in pixels of a reported error bar in Figure 3 of Kusner et al. (2015).

    Attributes
    ----------
    standard_error : scalar
        An estimate of the standard error of the mean of a binomial trial.
    """
    def __init__(self, test_error_height, error_bar_height):
        hundred_percent_height = 122.3581549180 / 70 * 100
        self._accuracy = 1 - (test_error_height / hundred_percent_height)
        self.standard_error = error_bar_height / 2.0 / hundred_percent_height * sqrt(5)

    def accuracy(self, significance_level=0.05):
        """Returns pointwise and interval estimates for the accuracy.

        We invoke the central limit theorem and assume that the sampling distribution
        of the mean is normal.

        Parameters
        ----------
        significance_level : scalar
            The likelihood that the actual accuracy falls into the
            confidence interval.

        Returns
        -------
        pointwise_estimate : scalar
            An unbiased pointwise estimate of the expected value of
            the accuracy.
        lower_bound : scalar
            The lower bound of the confidence interval for the accuracy.
        upper_bound : scalar
            The upper bound of the confidence interval for the accuracy.
        """

        pointwise_estimate = self._accuracy
        interval_radius = self.standard_error * scipy.stats.norm.ppf(1 - significance_level / 2.0)
        lower_bound = max(0.0, pointwise_estimate - interval_radius)
        upper_bound = min(1.0, pointwise_estimate + interval_radius)
        return (pointwise_estimate, lower_bound, upper_bound)
