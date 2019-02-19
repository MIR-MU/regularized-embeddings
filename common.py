from functools import total_ordering
import logging
import lzma
from math import sqrt
import pickle
from re import compile, match
import subprocess

import numpy as np
import scipy.stats
import sklearn.metrics

LOGGER = logging.getLogger(__name__)


def benjamini_hochberg(p_values):
    """Adjusts p-values from independent hypothesis tests to q-values.

    The q-values are determined using the false discovery rate (FDR) controlling procedure of
    Benjamini and Hochberg [BenjaminiHochberg1995]_.

    .. [BenjaminiHochberg1995] Benjamini, Yoav; Hochberg, Yosef (1995). "Controlling the false
       discovery rate: a practical and powerful approach to multiple testing". Journal of the
       Royal Statistical Society, Series B. 57 (1): 289–300. MR 1325392.

    Notes
    -----
    This method was adapted from `code posted to Stack Overflow by Eric Talevich`_.

    .. _code posted to Stack Overflow by Eric Talevich: https://stackoverflow.com/a/33532498/657401

    Parameters
    ----------
    p_values : iterable of scalar
        p-values from independent hypothesis tests.

    Returns
    -------
    q_values : iterable of scalar
        The p-values adjusted using the FDR controlling procedure.
    """

    p_value_array = np.asfarray(p_values)
    num_pvalues = len(p_value_array)
    descending_order = p_value_array.argsort()[::-1]
    original_order = descending_order.argsort()
    steps = num_pvalues / np.arange(num_pvalues, 0, -1)
    descending_q_values = np.minimum.accumulate(steps * p_value_array[descending_order]).clip(0, 1)
    q_values = descending_q_values[original_order]
    return q_values


def f_test(result_pairs, significance_level=0.05):
    """Performs the F-test on pairs of classification results.

    We invoke the central limit theorem and assume that the sampling
    distribution of the mean is normal.

    Parameters
    ----------
    result_pairs : iterable of tuple of {ClassificationResult,KusnerEtAlClassificationResult}
        Pairs of classification results.
    significance_level : scalar
        The likelihood that the population speed falls into the confidence
        interval.

    Returns
    -------
    test_results : list of bool
        The test results for the individual result pairs, where the value of
        ``True`` corresponds to a rejected hypothesis.
    """

    t_p_values = []
    for results in result_pairs:
        means = []
        variances = []
        nums_trials = []
        for result in results:
            if isinstance(result, ClassificationResult):
                num_successes = np.diag(result.confusion_matrix).sum()
                num_trials = np.sum(result.confusion_matrix)
                mean = num_successes / num_trials
                variance = mean * (1.0 - mean) / num_trials
            elif isinstance(result, KusnerEtAlClassificationResult):
                variance = result.standard_error**2
                num_trials = result.num_trials
                mean = result.accuracy()[0]
            means.append(mean)
            variances.append(variance)
            nums_trials.append(num_trials)
        f = max(variances) / min(variances)
        df1 = nums_trials[0] - 1
        df2 = nums_trials[1] - 1
        df = df1 + df2
        f_p_value = 1 - scipy.stats.f.cdf(f, df1, df2)
        if f_p_value < significance_level / 2.0:
            t = abs(means[0] - means[1]) / sqrt(
                (variances[0] / nums_trials[0]) + (variances[1] / nums_trials[1])
            )
        else:
            t = abs(means[0] - means[1]) / sqrt(
                (df1 * variances[0] + df2 * variances[1]) /
                df * sum(nums_trials) / (nums_trials[0] * nums_trials[1])
            )
        t_p_value = 1 - scipy.stats.t.cdf(t, df)
        t_p_values.append(t_p_value)
    t_q_values = benjamini_hochberg(t_p_values)
    test_results = [t_q_value < significance_level / 2.0 for t_q_value in t_q_values]
    return test_results


def read_speeds(results, significance_level=0.05):
    """Returns pointwise and interval estimates for the document processing speed.

    We invoke the central limit theorem and assume that the sampling
    distribution of the mean is normal.  We use the following method to `estimate
    the population variance from a set of means
    <https://stats.stackexchange.com/a/25079/116294>`_.

    Parameters
    ----------
    results : iterable of {ClassificationResult,KusnerEtAlClassificationResult}
        Classification results.
    significance_level : scalar
        The likelihood that the population speed falls into the confidence
        interval.

    Returns
    -------
    pointwise_estimate : scalar
        An unbiased pointwise estimate of the expected value of the speed.
    lower_bound : scalar
        The lower bound of the confidence interval for the speed.
    upper_bound : scalar
        The upper bound of the confidence interval for the speed.
    """

    matrix_production_re = compile(r'(Performed SVD in|Spent) (?P<duration>[^ ]*) seconds')
    similarity_speed_re = compile(r'Processed (?P<num_documents>[^ ]*) document pairs / (?P<duration>[^ ]*) seconds')
    matrix_production_duration = 0.0
    speeds = []
    nums_similarities = []
    similarity_durations = []
    for result in results:
        for line in result.params['speed_logs']:
            matrix_production_match = match(matrix_production_re, line)
            similarity_speed_match = match(similarity_speed_re, line)
            if matrix_production_match:
                matrix_production_duration = float(matrix_production_match.group('duration'))
            elif similarity_speed_match:
                num_similarities = int(similarity_speed_match.group('num_documents'))
                similarity_duration = float(similarity_speed_match.group('duration')) - matrix_production_duration
                matrix_production_duration = 0.0
                nums_similarities.append(num_similarities)
                similarity_durations.append(similarity_duration)

    pointwise_estimate = sum(nums_similarities) / sum(similarity_durations)
    speeds = np.divide(nums_similarities, similarity_durations)
    sample_weights = np.divide(nums_similarities, len(speeds) - 1)
    weighted_sample_variance = np.sum(np.multiply(sample_weights, np.subtract(speeds, pointwise_estimate)**2))
    standard_error_of_mean = sqrt(weighted_sample_variance / sum(nums_similarities))
    interval_radius = standard_error_of_mean * scipy.stats.norm.ppf(1 - significance_level / 2.0)
    lower_bound = max(0.0, pointwise_estimate - interval_radius)
    upper_bound = pointwise_estimate + interval_radius

    return (pointwise_estimate, lower_bound, upper_bound)


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

        We assume that the trials are Bernoulli.

        Parameters
        ----------
        significance_level : scalar
            The likelihood that the population accuracy falls into the
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
    num_trials : int
        The number of Bernoulli trials in the result.
    params : dict
        A dict of params related to the classification result.

    Attributes
    ----------
    num_trials : int
        The number of Bernoulli trials in the result.
    params : dict
        A dict of params related to the classification result.
    standard_error : scalar
        An estimate of the standard error of the mean of a Bernoulli trial.
    """
    def __init__(self, test_error_height, error_bar_height, num_trials, params):
        hundred_percent_height = 122.3581549180 / 70 * 100
        self._accuracy = 1 - (test_error_height / hundred_percent_height)
        self.num_trials = num_trials
        self.params = dict(params)
        self.standard_error = error_bar_height / 2.0 / hundred_percent_height * sqrt(5)

    def accuracy(self, significance_level=0.05):
        """Returns pointwise and interval estimates for the accuracy.

        We invoke the central limit theorem and assume that the sampling distribution
        of the mean is normal.

        Parameters
        ----------
        significance_level : scalar
            The likelihood that the population accuracy falls into the
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
