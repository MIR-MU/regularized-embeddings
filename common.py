from contextlib import contextmanager
import csv
from functools import total_ordering, reduce
from glob import glob
from itertools import chain, product, repeat
import json
import logging
import lzma
from math import sqrt
from multiprocessing import Pool
import operator
import os.path
import pickle
import random
import subprocess
from time import time

from gensim.corpora import Dictionary
from gensim.matutils import corpus2csc, unitvec
from gensim.models import KeyedVectors, TfidfModel, WordEmbeddingSimilarityIndex
from gensim.similarities import SparseTermSimilarityMatrix
from gensim.summarization.bm25 import BM25
from gensim.utils import tokenize
import nltk
from nltk.corpus import reuters
import numpy as np
from scipy import sparse
import scipy.stats
from sklearn.datasets import fetch_20newsgroups
import sklearn.metrics
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import train_test_split
import sklearn.preprocessing as preprocessing
from sparsesvd import sparsesvd
from tqdm import tqdm
from pyemd import emd

LOGGER = logging.getLogger(__name__)


def load_twitter():
    """Produces the training, validation, and test sets from TWITTER.

    Returns
    -------
    train : Dataset
        The training set.
    validation : Dataset
        The validation set.
    test : Dataset
        The test set.
    """

    try:
        twitter_train = Dataset.from_file('twitter_train')
        twitter_validation = Dataset.from_file('twitter_validation')
        twitter_test = Dataset.from_file('twitter_test')
    except IOError:
        make('TWITTER')
        twitter_X = []
        twitter_y = []
        category_names = ('positive', 'neutral', 'negative', 'irrelevant')
        with open('TWITTER/full-corpus.csv', 'rt') as f:
            reader = csv.DictReader(f)
            for line in reader:
                category_name = line['Sentiment']
                assert category_name in category_names
                category_number = category_names.index(category_name)
                document = line['TweetText']
                if category_name != 'irrelevant':
                    twitter_X.append(document)
                    twitter_y.append(category_number)

        (
            twitter_train_and_validation_X,
            twitter_test_X,
            twitter_train_and_validation_y,
            twitter_test_y,
        ) = train_test_split(
            twitter_X,
            twitter_y,
            train_size=2176,
            test_size=932,
            shuffle=True,
            random_state=42,
        )
        twitter_test = Dataset.from_documents(twitter_test_X, 'twitter_test', twitter_test_y)
        twitter_test.to_file()
        del twitter_X, twitter_y
        del twitter_test_X, twitter_test_y

        (
            twitter_train_X,
            twitter_validation_X,
            twitter_train_y,
            twitter_validation_y,
        ) = train_test_split(
            twitter_train_and_validation_X,
            twitter_train_and_validation_y,
            train_size=0.8,
            shuffle=False,
        )
        twitter_train = Dataset.from_documents(twitter_train_X, 'twitter_train', twitter_train_y)
        twitter_train.to_file()
        twitter_validation = Dataset.from_documents(twitter_validation_X, 'twitter_validation', twitter_validation_y)
        twitter_validation.to_file()
        del twitter_train_and_validation_X, twitter_train_and_validation_y
        del twitter_train_X, twitter_train_y
        del twitter_validation_X, twitter_validation_y

    return twitter_train, twitter_validation, twitter_test


def reuters_read_file_worker(args):
    category_number, fileid = args
    local_reuters_train_and_validation_X = []
    local_reuters_train_and_validation_y = []
    local_reuters_test_X = []
    local_reuters_test_y = []
    document = reuters.raw(fileid)
    if 'training' in fileid:
        local_reuters_train_and_validation_X.append(document)
        local_reuters_train_and_validation_y.append(category_number)
    else:
        local_reuters_test_X.append(document)
        local_reuters_test_y.append(category_number)
    return (
        local_reuters_train_and_validation_X,
        local_reuters_train_and_validation_y,
        local_reuters_test_X,
        local_reuters_test_y,
    )


def load_reuters():
    """Produces the training, validation, and test sets from REUTERS.

    Returns
    -------
    train : Dataset
        The training set.
    validation : Dataset
        The validation set.
    test : Dataset
        The test set.
    """

    try:
        reuters_train = Dataset.from_file('reuters_train')
        reuters_validation = Dataset.from_file('reuters_validation')
        reuters_test = Dataset.from_file('reuters_test')
    except IOError:
        nltk.download('reuters')
        categories = chain(
            *(
                zip(
                    repeat(category_number),
                    (
                        fileid
                        for fileid in reuters.fileids(category_name)
                        if len(reuters.categories(fileid)) == 1
                    )
                )
                for category_number, category_name in enumerate((
                    'acq', 'crude', 'earn', 'grain', 'interest',  'money-fx', 'ship', 'trade'
                ))
            )
        )
        reuters_train_and_validation_X = []
        reuters_train_and_validation_y = []
        reuters_test_X = []
        reuters_test_y = []
        with Pool(None) as pool:
            for (
                local_reuters_train_and_validation_X,
                local_reuters_train_and_validation_y,
                local_reuters_test_X,
                local_reuters_test_y,
            ) in pool.map(reuters_read_file_worker, categories):
                reuters_train_and_validation_X.extend(local_reuters_train_and_validation_X)
                reuters_train_and_validation_y.extend(local_reuters_train_and_validation_y)
                reuters_test_X.extend(local_reuters_test_X)
                reuters_test_y.extend(local_reuters_test_y)

        reuters_train_and_validation_X = reuters_train_and_validation_X[:5485]
        reuters_train_and_validation_y = reuters_train_and_validation_y[:5485]
        reuters_test_X = reuters_test_X[:2189]
        reuters_test_y = reuters_test_y[:2189]

        reuters_test = Dataset.from_documents(reuters_test_X, 'reuters_test', reuters_test_y)
        reuters_test.to_file()
        del reuters_test_X, reuters_test_y

        (
            reuters_train_X,
            reuters_validation_X,
            reuters_train_y,
            reuters_validation_y,
        ) = train_test_split(
            reuters_train_and_validation_X,
            reuters_train_and_validation_y,
            train_size=0.8,
            shuffle=True,
            random_state=42,
        )
        reuters_train = Dataset.from_documents(reuters_train_X, 'reuters_train', reuters_train_y)
        reuters_train.to_file()
        reuters_validation = Dataset.from_documents(reuters_validation_X, 'reuters_validation', reuters_validation_y)
        reuters_validation.to_file()
        del reuters_train_and_validation_X, reuters_train_and_validation_y
        del reuters_train_X, reuters_train_y
        del reuters_validation_X, reuters_validation_y

    return reuters_train, reuters_validation, reuters_test


def ohsumed_read_file_worker(pathname):
    filename = os.path.basename(pathname)
    local_ohsumed_X = []
    local_ohsumed_y = []
    pathnames = [
        (category_number, pathname)
        for category_number in range(1, 11)
        for pathname in glob('OHSUMED/ohsumed-all/C{:02}/{}'.format(category_number, filename))
    ]
    if len(pathnames) == 1:
        category_number = pathnames[0][0]
        with open(pathname, 'rt') as f:
            local_ohsumed_X.append(f.read())
            local_ohsumed_y.append(category_number)
    return local_ohsumed_X, local_ohsumed_y


def load_ohsumed():
    """Produces the training, validation, and test sets from OHSUMED.

    Returns
    -------
    train : Dataset
        The training set.
    validation : Dataset
        The validation set.
    test : Dataset
        The test set.
    """

    try:
        ohsumed_train = Dataset.from_file('ohsumed_train')
        ohsumed_validation = Dataset.from_file('ohsumed_validation')
        ohsumed_test = Dataset.from_file('ohsumed_test')
    except IOError:
        make('OHSUMED')
        pathnames = sorted(
            set(
                pathname
                for category_number in range(1, 11)
                for pathname in glob('OHSUMED/ohsumed-all/C{:02}/*'.format(category_number))
            )
        )
        ohsumed_X = []
        ohsumed_y = []
        with Pool(None) as pool:
            for local_ohsumed_X, local_ohsumed_y in pool.map(ohsumed_read_file_worker, pathnames):
                ohsumed_X.extend(local_ohsumed_X)
                ohsumed_y.extend(local_ohsumed_y)

        (
            ohsumed_train_and_validation_X,
            ohsumed_test_X,
            ohsumed_train_and_validation_y,
            ohsumed_test_y,
        ) = train_test_split(
            ohsumed_X,
            ohsumed_y,
            train_size=3999,
            test_size=5153,
            shuffle=True,
            random_state=42,
        )
        ohsumed_test = Dataset.from_documents(ohsumed_test_X, 'ohsumed_test', ohsumed_test_y)
        ohsumed_test.to_file()
        del ohsumed_X, ohsumed_y
        del ohsumed_test_X, ohsumed_test_y

        (
            ohsumed_train_X,
            ohsumed_validation_X,
            ohsumed_train_y,
            ohsumed_validation_y,
        ) = train_test_split(
            ohsumed_train_and_validation_X,
            ohsumed_train_and_validation_y,
            train_size=0.8,
            shuffle=False,
        )
        ohsumed_train = Dataset.from_documents(ohsumed_train_X, 'ohsumed_train', ohsumed_train_y)
        ohsumed_train.to_file()
        ohsumed_validation = Dataset.from_documents(ohsumed_validation_X, 'ohsumed_validation', ohsumed_validation_y)
        ohsumed_validation.to_file()
        del ohsumed_train_and_validation_X, ohsumed_train_and_validation_y
        del ohsumed_train_X, ohsumed_train_y
        del ohsumed_validation_X, ohsumed_validation_y

    return ohsumed_train, ohsumed_validation, ohsumed_test


def bbcsport_read_file_worker(args):
    category_number, filename = args
    local_bbcsport_X = []
    local_bbcsport_y = []
    with open(filename, 'rt') as f:
        local_bbcsport_X.append(f.read())
        local_bbcsport_y.append(category_number)
    return local_bbcsport_X, local_bbcsport_y


def load_bbcsport():
    """Produces the training, validation, and test sets from BBCSport.

    Returns
    -------
    train : Dataset
        The training set.
    validation : Dataset
        The validation set.
    test : Dataset
        The test set.
    """

    try:
        bbcsport_train = Dataset.from_file('bbcsport_train')
        bbcsport_validation = Dataset.from_file('bbcsport_validation')
        bbcsport_test = Dataset.from_file('bbcsport_test')
    except IOError:
        make('BBC')
        categories = chain(
            *(
                zip(
                    repeat(category_number),
                    glob('BBC/bbcsport/{}/*.txt'.format(category_name))
                )
                for category_number, category_name in enumerate((
                    'athletics', 'cricket', 'football', 'rugby', 'tennis',
                ))
            )
        )
        bbcsport_X = []
        bbcsport_y = []
        with Pool(None) as pool:
            for local_bbcsport_X, local_bbcsport_y in pool.map(bbcsport_read_file_worker, categories):
                bbcsport_X.extend(local_bbcsport_X)
                bbcsport_y.extend(local_bbcsport_y)

        (
            bbcsport_train_and_validation_X,
            bbcsport_test_X,
            bbcsport_train_and_validation_y,
            bbcsport_test_y,
        ) = train_test_split(
            bbcsport_X,
            bbcsport_y,
            train_size=517,
            test_size=220,
            shuffle=True,
            random_state=42,
        )
        bbcsport_test = Dataset.from_documents(
            bbcsport_test_X,
            'bbcsport_test',
            bbcsport_test_y,
        )
        bbcsport_test.to_file()
        del bbcsport_X, bbcsport_y
        del bbcsport_test_X, bbcsport_test_y

        (
            bbcsport_train_X,
            bbcsport_validation_X,
            bbcsport_train_y,
            bbcsport_validation_y,
        ) = train_test_split(
            bbcsport_train_and_validation_X,
            bbcsport_train_and_validation_y,
            train_size=0.8,
            shuffle=False,
        )
        bbcsport_train = Dataset.from_documents(
            bbcsport_train_X,
            'bbcsport_train',
            bbcsport_train_y,
        )
        bbcsport_train.to_file()
        bbcsport_validation = Dataset.from_documents(
            bbcsport_validation_X,
            'bbcsport_validation',
            bbcsport_validation_y,
        )
        bbcsport_validation.to_file()
        del bbcsport_train_and_validation_X, bbcsport_train_and_validation_y
        del bbcsport_train_X, bbcsport_train_y
        del bbcsport_validation_X, bbcsport_validation_y

    return bbcsport_train, bbcsport_validation, bbcsport_test


def bbc_read_file_worker(args):
    category_number, filename = args
    local_bbc_X = []
    local_bbc_y = []
    with open(filename, 'rt') as f:
        local_bbc_X.append(f.read())
        local_bbc_y.append(category_number)
    return local_bbc_X, local_bbc_y


def load_bbc():
    """Produces the training, validation, and test sets from BBC.

    Returns
    -------
    train : Dataset
        The training set.
    validation : Dataset
        The validation set.
    test : Dataset
        The test set.
    """

    try:
        bbc_train = Dataset.from_file('bbc_train')
        bbc_validation = Dataset.from_file('bbc_validation')
        bbc_test = Dataset.from_file('bbc_test')
    except IOError:
        make('BBC')
        categories = chain(
            *(
                zip(
                    repeat(category_number),
                    glob('BBC/bbc/{}/*.txt'.format(category_name))
                )
                for category_number, category_name in enumerate((
                    'business', 'entertainment', 'politics', 'sport', 'tech',
                ))
            )
        )
        bbc_X = []
        bbc_y = []
        with Pool(None) as pool:
            for local_bbc_X, local_bbc_y in pool.map(bbc_read_file_worker, categories):
                bbc_X.extend(local_bbc_X)
                bbc_y.extend(local_bbc_y)

        (
            bbc_train_and_validation_X,
            bbc_test_X,
            bbc_train_and_validation_y,
            bbc_test_y,
        ) = train_test_split(
            bbc_X,
            bbc_y,
            train_size=0.7,
            shuffle=True,
            random_state=42,
        )
        bbc_test = Dataset.from_documents(bbc_test_X, 'bbc_test', bbc_test_y)
        bbc_test.to_file()
        del bbc_X, bbc_y
        del bbc_test_X, bbc_test_y

        (
            bbc_train_X,
            bbc_validation_X,
            bbc_train_y,
            bbc_validation_y,
        ) = train_test_split(
            bbc_train_and_validation_X,
            bbc_train_and_validation_y,
            train_size=0.8,
            shuffle=False,
        )
        bbc_train = Dataset.from_documents(bbc_train_X, 'bbc_train', bbc_train_y)
        bbc_train.to_file()
        bbc_validation = Dataset.from_documents(bbc_validation_X, 'bbc_validation', bbc_validation_y)
        bbc_validation.to_file()
        del bbc_train_and_validation_X, bbc_train_and_validation_y
        del bbc_train_X, bbc_train_y
        del bbc_validation_X, bbc_validation_y

    return bbc_train, bbc_validation, bbc_test


def amazon_read_file_worker(args):
    category_number, filename = args
    local_amazon_X = []
    local_amazon_y = []
    with open(filename, 'rt') as f:
        for line_str in f:
            line = json.loads(line_str)
            review_text = line['reviewText']
            local_amazon_X.append(review_text)
            local_amazon_y.append(category_number)
    return (local_amazon_X, local_amazon_y)


def load_amazon():
    """Produces the training, validation, and test sets from AMAZON.

    Returns
    -------
    train : Dataset
        The training set.
    validation : Dataset
        The validation set.
    test : Dataset
        The test set.
    """
    try:
        amazon_train = Dataset.from_file('amazon_train')
        amazon_validation = Dataset.from_file('amazon_validation')
        amazon_test = Dataset.from_file('amazon_test')
    except IOError:
        make('AMAZON')
        categories = chain(
            *(
                zip(
                    repeat(category_number),
                    glob('AMAZON/reviews_{}_5.json.gz_split*'.format(category_name))
                )
                for category_number, category_name in enumerate((
                    'Books', 'CDs_and_Vinyl', 'Electronics', 'Home_and_Kitchen',
                    # 'Amazon_Instant_Video', 'Apps_for_Android', 'Automotive', 'Baby', 'Beauty',
                    # 'Cell_Phones_and_Accessories', 'Clothing_Shoes_and_Jewelry',
                    # 'Digital_Music', 'Grocery_and_Gourmet_Food', 'Health_and_Personal_Care',
                    # 'Kindle_Store', 'Movies_and_TV', 'Musical_Instruments',
                    # 'Office_Products', 'Patio_Lawn_and_Garden', 'Pet_Supplies','Sports_and_Outdoors',
                    # 'Tools_and_Home_Improvement', 'Toys_and_Games', 'Video_Games',
                ))
            )
        )
        amazon_X = []
        amazon_y = []
        for local_amazon_X, local_amazon_y in map(amazon_read_file_worker, categories):
            amazon_X.extend(local_amazon_X)
            amazon_y.extend(local_amazon_y)

        (
            amazon_train_and_validation_X,
            amazon_test_X,
            amazon_train_and_validation_y,
            amazon_test_y,
        ) = train_test_split(
            amazon_X,
            amazon_y,
            train_size=5600,
            test_size=2400,
            shuffle=True,
            random_state=42,
        )
        amazon_test = Dataset.from_documents(amazon_test_X, 'amazon_test', amazon_test_y)
        amazon_test.to_file()
        del amazon_X, amazon_y
        del amazon_test_X, amazon_test_y

        (
            amazon_train_X,
            amazon_validation_X,
            amazon_train_y,
            amazon_validation_y,
        ) = train_test_split(
            amazon_train_and_validation_X,
            amazon_train_and_validation_y,
            train_size=0.8,
            shuffle=False,
        )
        amazon_train = Dataset.from_documents(amazon_train_X, 'amazon_train', amazon_train_y)
        amazon_train.to_file()
        amazon_validation = Dataset.from_documents(amazon_validation_X, 'amazon_validation', amazon_validation_y)
        amazon_validation.to_file()
        del amazon_train_and_validation_X, amazon_train_and_validation_y
        del amazon_train_X, amazon_train_y
        del amazon_validation_X, amazon_validation_y

    return amazon_train, amazon_validation, amazon_test


def load_20news():
    """Produces the training, validation, and test sets from 20NEWS.

    Returns
    -------
    train : Dataset
        The training set.
    validation : Dataset
        The validation set.
    test : Dataset
        The test set.
    """

    try:
        newsgroups_train = Dataset.from_file('newsgroups_train')
        newsgroups_validation = Dataset.from_file('newsgroups_validation')
        newsgroups_test = Dataset.from_file('newsgroups_test')
    except IOError:
        newsgroups_train_and_validation_raw = fetch_20newsgroups(subset='train')
        newsgroups_train_and_validation_X = newsgroups_train_and_validation_raw.data[:11293]
        newsgroups_train_and_validation_y = newsgroups_train_and_validation_raw.target[:11293]
        del newsgroups_train_and_validation_raw

        newsgroups_test_raw = fetch_20newsgroups(subset='test')
        newsgroups_test_X = newsgroups_test_raw.data[:7528]
        newsgroups_test_y = newsgroups_test_raw.target[:7528]
        del newsgroups_test_raw

        (
            newsgroups_train_X,
            newsgroups_validation_X,
            newsgroups_train_y,
            newsgroups_validation_y
        ) = train_test_split(
            newsgroups_train_and_validation_X,
            newsgroups_train_and_validation_y,
            train_size=0.8,
            shuffle=True,
            random_state=42,
        )
        newsgroups_train = Dataset.from_documents(
            newsgroups_train_X,
            'newsgroups_train',
            newsgroups_train_y,
        )
        newsgroups_train.to_file()
        newsgroups_validation = Dataset.from_documents(
            newsgroups_validation_X,
            'newsgroups_validation',
            newsgroups_validation_y,
        )
        newsgroups_validation.to_file()
        del newsgroups_train_and_validation_X, newsgroups_train_and_validation_y
        del newsgroups_train_X, newsgroups_train_y
        del newsgroups_validation_X, newsgroups_validation_y

        newsgroups_test = Dataset.from_documents(
            newsgroups_test_X,
            'newsgroups_test',
            newsgroups_test_y,
        )
        newsgroups_test.to_file()
        del newsgroups_test_X, newsgroups_test_y

    return newsgroups_train, newsgroups_validation, newsgroups_test


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


def grid_search(grid_specification):
    """Performs a grid search.

    Parameters
    ----------
    grid_specification : dict of (object, iterable)
        A specification of the dimensions and the possible
        values of the individual parameters.

    Yields
    ------
    grid_params : dict of (object, object)
        A single position in the grid. An empty dict is yielded
        for an empty grid.
    """

    if not grid_specification:
        yield dict()
    else:
        keys, iterables = zip(*grid_specification.items())
        for grid_params in product(*iterables):
            yield dict(zip(keys, grid_params))


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


@contextmanager
def log_speed(speed_logs, message):
    """Measures and logs the duration of a context.

    Parameters
    ----------
    speed_logs : list of str
        Text logs regarding speed.
    message : str
        A log message, which will be processed using ``string.format()``
        with the duration of a context as the only parameter.
    """
    start_time = time()
    yield
    stop_time = time()
    duration = stop_time - start_time
    speed_log = message.format(duration)
    speed_logs.append(speed_log)
    LOGGER.info(speed_log)


def cached_sparsesvd(basename, speed_logs, *args):
    """Produces an SVD of a document matrix, loading it if cached.

    Parameters
    ----------
    basename : str
        The basename of the cached SVD matrix.
    speed_logs : list of str
        Text logs regarding the processing speed.
    args : iterable
        The arguments of the `sparsesvd` function.

    Returns
    -------
    ut : numpy.ndarray
        The :math:`U^T` matrix.
    s : numpy.ndarray
        The :math:`S` matrix.
    vt : numpy.ndarray
        The :math:`V^T` matrix.
    """

    with log_speed(speed_logs, 'Spent {} seconds producing an SVD matrix'):
        make('matrices')
        filename = 'matrices/svd-{}.pkl.xz'.format(basename)
        try:
            with lzma.open(filename, 'rb') as f:
                LOGGER.debug('Loading SVD matrices from file {}.'.format(filename))
                ut, s, vt = pickle.load(f)
        except IOError:
            with log_speed(speed_logs, 'Performed SVD in {} seconds'):
                ut, s, vt = sparsesvd(*args)
            with lzma.open(filename, 'wb', preset=0) as f:
                LOGGER.info('Saving SVD matrices to file {}.'.format(filename))
                pickle.dump((ut, s, vt), f, 4)
        return (ut, s, vt)


def cached_sparse_term_similarity_matrix(basename, speed_logs, *args, **kwargs):
    """Produces a sparse term similarity matrix, loading it if cached.

    Parameters
    ----------
    basename : str
        The basename of the cached SVD matrix.
    speed_logs : list of str
        Text logs regarding the processing speed.
    args : iterable
        The arguments of the `SparseTermSimilarityMatrix` constructor.
    kwargs : dict
        The keyword arguments of the `SparseTermSimilarityMatrix` constructor.

    Returns
    -------
    term_matrix : gensim.similarities.SparseTermSimilarityMatrix
        The sparse term similarity matrix.
    """

    with log_speed(speed_logs, 'Spent {} seconds producing a term similarity matrix'):
        make('matrices')
        filename = 'matrices/termsim-{}.pkl.xz'.format(basename)
        try:
            with lzma.open(filename, 'rb') as f:
                LOGGER.debug('Loading term similarity matrix from file {}.'.format(filename))
                term_matrix = pickle.load(f)
        except IOError:
            with log_speed(speed_logs, 'Constructed term similarity matrix in {} seconds'):
                term_sims = SparseTermSimilarityMatrix(*args, **kwargs)
            term_matrix = term_sims.matrix
            with lzma.open(filename, 'wb', preset=0) as f:
                LOGGER.info('Saving term similarity matrix to file {}.'.format(filename))
                pickle.dump(term_matrix, f, 4)
        return term_matrix


def inverse_wmd_worker(args):
    """Produces inverse word mover's distances for a collection document and a query corpus.

    Parameters
    ----------
    (row_number, query_document) : (int, list of (int, float))
        The identifier of a query document and the query document.
    (collection_number, collection_document) : (int, list of (int, float))
        The identifier of a collection document and the collection document.
    num_bits : int
        The quantization level of the word vectors used to compute the word mover's distance.

    Returns
    -------
    row_number : int
        The identifier of the query document.
    column_number : int
        The identifier of the collection document.
    inverse_distance : float
        The inverse word mover's distances between the collection document and the query document.
    """

    (row_number, query_document), (column_number, collection_document), num_bits = args
    embedding_matrix = common_embedding_matrices[num_bits]
    embedding_matrix_norm_squared = common_embedding_matrices_norm_squared[num_bits]
    query_document = dict(query_document)
    collection_document = dict(collection_document)
    shared_terms = tuple(set(query_document.keys()) | set(collection_document.keys()))
    if shared_terms:
        translated_query_document = np.array(list(map(
            lambda x: query_document.get(x, 0.0),
            shared_terms,
        )), dtype=float)
        translated_collection_document = np.array(list(map(
            lambda x: collection_document.get(x, 0.0),
            shared_terms,
        )), dtype=float)
        shared_embedding_matrix = embedding_matrix[shared_terms, :].astype(float)
        shared_embedding_matrix_norm_squared = embedding_matrix_norm_squared[shared_terms, :].astype(float)
        distance_matrix = euclidean_distances(
            shared_embedding_matrix,
            X_norm_squared=shared_embedding_matrix_norm_squared,
        )
        distance = emd(translated_collection_document, translated_query_document, distance_matrix)
        if distance == 0.0:
            inverse_distance = float('inf')
        else:
            inverse_distance = 1.0 / distance
    else:
        inverse_distance = 0.0
    return (row_number, column_number, inverse_distance)


def bm25_worker(args):
    """Transform a document using Okapi BM25 term weighting.

    Parameters
    ----------
    document : list of list of (int, float)
        A document.
    k1 : float
        The :math:`k_1` parameter of Okapi BM25.
    b : float
        The :math:`b` parameter of Okapi BM25.
    bm25 : gensim.summarization.bm25.BM25
        Statistics about the documents in a dataset.
    doc_len : int
        The length of the document.
    dictionary : gensim.corpora.Dictionary
        A dictionary used in a dataset.

    Returns
    -------
    bm25_document : list of list of (int, float)
        The transformed document.
    """

    document, k1, b, bm25, doc_len, dictionary = args
    bm25_document = [
        (
            term_id,
            (
                bm25.idf[dictionary[term_id]] * term_weight * (k1 + 1) / (
                    term_weight + k1 * (1 - b + b * doc_len / bm25.avgdl)
                )
            )
        )
        for term_id, term_weight in document
    ]
    return bm25_document


def binarize_worker(document):
    """Binarizes a BOW document.

    Parameters
    ----------
    document : list of (int, float)
        A document.

    Returns
    -------
    binarized_document : list of (int, float)
        The binarized document.
    """

    binarized_document = [(term_id, 1) for term_id, _ in document]
    return binarized_document


def pivot_worker(args):
    """Pivots a BOW document using the b SMART scheme.

    Parameters
    ----------
    document : list of list of (int, float)
        A document.
    slope : float
        The pivoting slope.

    Returns
    -------
    pivoted_document : list of list of (int, float)
        The pivoted document.
    """

    document, slope, avgdl = args
    doclen = sum(len(token) for token in document)

    pivoted_document = [
        (
            term_id,
            term_weight / (
                1.0 - slope + slope * (doclen / avgdl)
            ),
        )
        for term_id, term_weight in document
    ]
    return pivoted_document


def translate_embeddings(embeddings, dictionary):
    """Translates word embeddings into a word embedding matrix using a dictionary.

    Parameters
    ----------
    embeddings : gensim.similarities.KeyedVectors
        Word embeddings.
    dictionary : gensim.corpora.Dictionary
        A dictionary that specifies the order of words in an embedding matrix.

    Returns
    -------
    embedding_matrix : numpy.ndarray
        An embedding matrix. Embeddings for words that are not in the
        dictionary are not included in the matrix. Rows corresponding to words
        with no embeddings are filled with zeros.
    """

    source_matrix = embeddings.vectors
    target_dtype = source_matrix.dtype
    target_shape = (len(dictionary), source_matrix.shape[1])
    target_matrix = np.zeros(target_shape, dtype=target_dtype)
    source_rows, target_rows = zip(*(
        (embeddings.vocab[dictionary[term_id]].index, term_id)
        for term_id in sorted(dictionary.keys())
        if dictionary[term_id] in embeddings.vocab
    ))
    target_matrix[target_rows, :] = source_matrix[source_rows, :]
    return target_matrix


def translate_document_worker(args):
    """Translates a BOW document from a source dictionary to a target dictionary.

    Parameters
    ----------
    document : list of list of (int, float)
        A document in the bag of words (BOW) representation.
    source_dictionary : gensim.corpora.Dictionary
        The source dictionary.
    target_dictionary : gensim.corpora.Dictionary
        The target dictionary.

    Returns
    -------
    translated_document : list of list of (int, float)
        The translated document.

    """

    document, source_dictionary, target_dictionary = args
    translated_document = [
        (target_dictionary.token2id[source_dictionary[term_id]], term_weight)
        for term_id, term_weight in document
        if source_dictionary[term_id] in target_dictionary.token2id
    ]
    return translated_document


def tokenize_worker(document):
    """Tokenizes a single document.

    Parameters
    ----------
    document : str
        An untokenized document.

    Returns
    -------
    tokenized_document : list of str
        The tokenized document.
    """

    tokenized_document = list(tokenize(document, lower=True))
    return tokenized_document


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


class Dataset(object):
    """A dataset with a dictionary, additional statistics, and document classes.

    Parameters
    ----------
    name : str
        The unique name of the dataset.
    bm25 : gensim.summarization.bm25.BM25
        Statistics about the documents in the dataset.
    corpus : list of list of str
        The tokenized corpus of documents.
    avgdl : float
        The average character length of a document.
    dictionary : gensim.corpora.Dictionary
        A mapping between tokens and token ids.
    target : {list, None}, optional
        The document classes.

    Attributes
    ----------
    name : str
        The unique name of the dataset.
    bm25 : gensim.summarization.bm25.BM25
        Statistics about the documents in the dataset.
    corpus : list of list of str
        The tokenized corpus of documents.
    avgdl : float
        The average character length of a document.
    dictionary : gensim.corpora.Dictionary
        A mapping between tokens and token ids.
    target : {list, None}
        The document classes. Defaults to `None`.
    """
    def __init__(self, name, bm25, corpus, avgdl, dictionary, target=None):
        self.name = name
        self.bm25 = bm25
        self.corpus = corpus
        self.avgdl = avgdl
        self.dictionary = dictionary
        self.target = target

    @staticmethod
    def from_documents(documents, name, target=None):
        """Loads a dataset from an untokenized corpus.

        Parameters
        ----------
        documents : iterable of str
            The untokenized corpus of documents.
        name : str
            The unique name of the dataset.
        target : {iterable, None}, optional
            The document classes. Defaults to `None`.

        Returns
        -------
        dataset : Dataset
            The dataset constructed from the untokenized corpus.
        """
        LOGGER.info('Reading dataset from untokenized corpus.')
        corpus = list(map(tokenize_worker, documents))
        bm25 = BM25(corpus)
        avgdl = sum(sum(len(token) for token in document) for document in corpus) / len(corpus)
        dictionary = Dictionary(corpus, prune_at=None)
        if target is not None:
            target = list(target)
        dataset = Dataset(name, bm25, corpus, avgdl, dictionary, target)
        return dataset

    @staticmethod
    def from_file(name):
        """Loads a dataset from a file.

        Parameters
        ----------
        name : str
            A unique name of the dataset.

        Returns
        -------
        dataset : Dataset
            The dataset loaded from the file.
        """
        filename = 'corpora/{}.pkl.xz'.format(name)
        with lzma.open(filename, 'rb') as f:
            LOGGER.info('Loading dataset from file {}.'.format(filename))
            kwargs = pickle.load(f)
            kwargs['name'] = name
            dataset = Dataset(**kwargs)
        return dataset

    def to_file(self):
        """Saves a dataset to a file.

        """
        name = self.name
        filename = 'corpora/{}.pkl.xz'.format(name)
        with lzma.open(filename, 'wb', preset=0) as f:
            LOGGER.info('Saving dataset to file {}.'.format(filename))
            pickle.dump(vars(self), f, 4)

    def classify(self, validation, test, space='vsm', weights='bow', measure='inner_product', num_bits=32):
        """Performs classification using this dataset as the training set.

        Parameters
        ----------
        validation : Dataset
            The validation set.
        test : Dataset
            The test set.
        space : {'random', 'vsm', 'sparse_soft_vsm', 'dense_soft_vsm', 'lsi'}, optional
            The document representation used for the classification.
        weights : {'random', 'binary', 'bow', 'tfidf', 'bm25'}, optional
            The term weighting scheme used for the classification.
        measure : {'random', 'inner_product', 'wmd'}, optional
            The similarity measure used for the classification.
        num_bits : {'random', 1, 32}, optional
            The number of bits used to construct Word2Bit embeddings.
        """
        params = {
            'space': space,
            'weights': weights,
            'measure': measure,
            'num_bits': num_bits,
            'task': 'classification',
            'speed_logs': [],
        }
        grid_specification = {}
        train = self

        if space == 'random':
            y_true = test.target
            classes = list(set(y_true))
            random.seed(42)
            y_pred = list(map(lambda _: random.choice(classes), range(len(y_true))))
            result = ClassificationResult.from_results(y_true, y_pred, params)
            return result

        if weights == 'tfidf':
            grid_specification.update({'slope': np.linspace(0, 1, 11)})
        elif weights == 'bm25':
            grid_specification.update({'k1': np.linspace(1.2, 2, 9)})

        if space == 'sparse_soft_vsm':
            grid_specification.update({
                'symmetric': (True, False),
                'positive_definite': (True, False),
                'tfidf': (True, False),
                'nonzero_limit': (100, 200, 300, 400, 500, 600),
            })

        LOGGER.info('Grid searching on dataset {} with params {}'.format(self.name, params))
        for grid_params in tqdm(
                    grid_search(grid_specification),
                    position=0,
                    total=reduce(operator.mul, (
                        len(values)
                        for values in grid_specification.values()
                    ), 1),
                ):
            params.update(grid_params)
            doc_sims = train.get_similarities(validation, params)
            results = []
            for k in range(1, 20):
                LOGGER.info('Finding k={} nearest neighbors'.format(k))
                params['k'] = k
                result = ClassificationResult.from_similarities(doc_sims, train, validation, params)
                results.append(result)
        best_result = max(results)

        params = best_result.params
        doc_sims = train.get_similarities(test, params)
        result = ClassificationResult.from_similarities(doc_sims, train, test, params)
        return result

    def get_similarities(self, queries, params):
        """Computes the similarities between two datasets.

        Parameters
        ----------
        queries : Dataset
            A dataset of queries.
        params : dict
            The parameters of the vector space model.

        Returns
        -------
        doc_similarities : np.matrix
            The similarities between the two datasets.
        """

        space = params['space']
        if space == 'sparse_soft_vsm':
            tfidf = params['tfidf']
            symmetric = params['symmetric']
            positive_definite = params['positive_definite']
            nonzero_limit = params['nonzero_limit']

        weights = params['weights']
        if weights == 'tfidf':
            slope = params['slope']
        elif weights == 'bm25':
            k1 = params['k1']

        task = params['task']
        measure = params['measure']
        num_bits = params['num_bits']
        speed_logs = params['speed_logs']

        collection = self
        num_document_pairs = len(collection.corpus) * len(queries.corpus)

        with log_speed(speed_logs, 'Processed {} document pairs / {{}} seconds'.format(num_document_pairs)):

            if weights == 'tfidf':
                if 'collection_corpus' not in params:
                    params['collection_corpus'] = list(map(collection.dictionary.doc2bow, collection.corpus))
                collection_corpus = params['collection_corpus']
                collection_tfidf = TfidfModel(dictionary=collection.dictionary, smartirs='dtn')
                collection_corpus = map(pivot_worker, zip(
                    collection_tfidf[collection_corpus],
                    repeat(slope),
                    repeat(collection.avgdl),
                ))
                collection_corpus = map(translate_document_worker, zip(
                    collection_corpus,
                    repeat(collection.dictionary),
                    repeat(common_dictionary),
                ))
            else:
                if 'collection_corpus' not in params:
                    params['collection_corpus'] = list(map(common_dictionary.doc2bow, collection.corpus))
                collection_corpus = params['collection_corpus']
                if weights == 'bow':
                    if space == 'dense_soft_vsm':
                        norm = 'l1'
                    else:
                        norm = 'l2'
                    collection_corpus = map(lambda document: unitvec(document, norm), collection_corpus)
                elif weights == 'binary':
                    collection_corpus = map(binarize_worker, collection_corpus)
                elif weights == 'bm25':
                    collection_corpus = map(bm25_worker, zip(
                        collection_corpus,
                        repeat(k1),
                        repeat(0.25),
                        repeat(collection.bm25),
                        collection.bm25.doc_len,
                        repeat(common_dictionary),
                    ))
            collection_corpus = list(collection_corpus)

            if weights == 'tfidf' and task == 'classification':
                if 'query_corpus' not in params:
                    params['query_corpus'] = list(map(collection.dictionary.doc2bow, queries.corpus))
                query_corpus = params['query_corpus']
                query_corpus = map(pivot_worker, zip(
                    collection_tfidf[query_corpus],
                    repeat(slope),
                    repeat(collection.avgdl),
                ))
                query_corpus = map(translate_document_worker, zip(
                    query_corpus,
                    repeat(collection.dictionary),
                    repeat(common_dictionary),
                ))
            else:
                if 'query_corpus' not in params:
                    params['query_corpus'] = list(map(common_dictionary.doc2bow, queries.corpus))
                query_corpus = params['query_corpus']
                if weights in ('binary', 'bm25'):
                    query_corpus = map(binarize_worker, query_corpus)
                elif weights == 'bow' and space != 'dense_soft_vsm':
                    query_corpus = map(unitvec, query_corpus)
            query_corpus = list(query_corpus)

            if measure == 'wmd':
                doc_sims = np.empty((len(query_corpus), len(collection_corpus)), dtype=float)
                with Pool(None) as pool:
                    for row_number, column_number, similarity in pool.imap_unordered(
                                inverse_wmd_worker,
                                tqdm(
                                    product(
                                        enumerate(query_corpus),
                                        enumerate(collection_corpus),
                                        (num_bits, ),
                                    ),
                                    position=1,
                                    total=len(query_corpus) * len(collection_corpus),
                                ),
                            ):
                        doc_sims[row_number, column_number] = similarity
            elif measure == 'inner_product':
                collection_matrix = corpus2csc(collection_corpus, len(common_dictionary))
                query_matrix = corpus2csc(query_corpus, len(common_dictionary))

                if space == 'vsm':
                    doc_sims = collection_matrix.T.dot(query_matrix).T.todense()
                elif space in ('lsi', 'dense_soft_vsm'):
                    if space == 'lsi':
                        if weights == 'tfidf':
                            weights_str = 'tfidf_{:.1}'.format(slope)
                        elif weights == 'bm25':
                            weights_str = 'bm25_{:.1}'.format(k1)
                        else:
                            weights_str = weights
                        lsi_basename = '{dataset_name}-{task}-{weights_str}'.format(
                            dataset_name=collection.name,
                            task=task,
                            weights_str=weights_str,
                        )
                        ut, s, vt = cached_sparsesvd(lsi_basename, speed_logs, collection_matrix, 500)
                        collection_matrix = vt
                        query_matrix = np.diag(1.0 / s).dot(scipy.sparse.csc_matrix.dot(ut, query_matrix))
                        del ut
                    elif space == 'dense_soft_vsm':
                        embedding_matrix = common_embedding_matrices[num_bits]
                        if weights == 'bow':
                            embedding_matrix = preprocessing.normalize(embedding_matrix, norm='l1')
                        elif weights == 'tfidf':
                            embedding_matrix = preprocessing.normalize(embedding_matrix, norm='l2')
                        collection_matrix = scipy.sparse.csc_matrix.dot(embedding_matrix.T, collection_matrix)
                        query_matrix = scipy.sparse.csc_matrix.dot(embedding_matrix.T, query_matrix)
                    if space != 'dense_soft_vsm' or weights != 'bow':
                        collection_matrix = preprocessing.normalize(collection_matrix.T, norm='l2').T
                        query_matrix = preprocessing.normalize(query_matrix.T, norm='l2').T
                    doc_sims = collection_matrix.T.dot(query_matrix).T
                elif space == 'sparse_soft_vsm':
                    term_basename = '{num_bits}-{tfidf}-{symmetric}-{positive_definite}-{nonzero_limit}'.format(
                        num_bits=num_bits,
                        tfidf=tfidf,
                        symmetric=symmetric,
                        positive_definite=positive_definite,
                        nonzero_limit=nonzero_limit,
                    )
                    term_index = WordEmbeddingSimilarityIndex(common_embeddings[num_bits])
                    term_matrix = cached_sparse_term_similarity_matrix(
                        term_basename,
                        speed_logs,
                        term_index,
                        common_dictionary,
                        tfidf=common_tfidf if tfidf else None,
                        symmetric=symmetric,
                        positive_definite=positive_definite,
                        nonzero_limit=nonzero_limit,
                    )
                    collection_matrix_norm = collection_matrix.T.dot(term_matrix).multiply(collection_matrix.T).sum(axis=1).T
                    query_matrix_norm = query_matrix.T.dot(term_matrix).multiply(query_matrix.T).sum(axis=1).T
                    collection_matrix = collection_matrix.multiply(sparse.csr_matrix(1 / np.sqrt(collection_matrix_norm)))
                    query_matrix = query_matrix.multiply(sparse.csr_matrix(1 / np.sqrt(query_matrix_norm)))
                    collection_matrix[collection_matrix == np.inf] = 0.0
                    query_matrix[query_matrix == np.inf] = 0.0
                    doc_sims = collection_matrix.T.dot(term_matrix).dot(query_matrix).T.todense()

        return doc_sims

#
# try:
#     common_corpus = Dataset.from_file('fil8')
# except IOError:
#     make('corpora')
#     with open('corpora/fil8', 'rt') as f:
#         common_corpus = Dataset.from_documents(f, 'fil8')
#         common_corpus.to_file()
# common_dictionary = common_corpus.dictionary
# common_tfidf = TfidfModel(dictionary=common_dictionary, smartirs='dtn')
#
# subprocess.call('make vectors', shell=True)
# common_embeddings = {
#     1: KeyedVectors.load_word2vec_format('vectors/1b_1000d_vectors_e10_nonbin', binary=False),
#     32: KeyedVectors.load_word2vec_format('vectors/32b_200d_vectors_e10_nonbin', binary=False),
# }
# common_embedding_matrices = {
#     num_bits: translate_embeddings(embeddings, common_dictionary)
#     for num_bits, embeddings in common_embeddings.items()
# }
# common_embedding_matrices_norm_squared = {
#     num_bits: (embedding_matrix**2).sum(axis=1)[:, np.newaxis]
#     for num_bits, embedding_matrix in common_embedding_matrices.items()
# }
