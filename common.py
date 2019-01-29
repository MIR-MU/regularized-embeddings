import csv
from functools import total_ordering
from glob import glob
from itertools import chain, product, repeat
import json
import logging
import lzma
from multiprocessing import Pool
import pickle
import subprocess

from gensim.corpora import Dictionary
from gensim.matutils import corpus2csc, unitvec
from gensim.models import TfidfModel, WordEmbeddingSimilarityIndex
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
from sklearn.model_selection import train_test_split
from sparsesvd import sparsesvd
from video699.common import binomial_confidence_interval

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
        subprocess.call('make TWITTER', shell=True)
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
            test_size=0.2,
            shuffle=True,
            random_state=42,
        )
        twitter_test = Dataset.from_documents(twitter_test_X, twitter_test_y, 'twitter_test')
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
            test_size=0.2,
            shuffle=False,
        )
        twitter_train = Dataset.from_documents(twitter_train_X, twitter_train_y, 'twitter_train')
        twitter_train.to_file()
        twitter_validation = Dataset.from_documents(twitter_validation_X, twitter_validation_y, 'twitter_validation')
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
                    reuters.fileids(category_name)
                )
                for category_number, category_name in enumerate(reuters.categories())
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
        reuters_test = Dataset.from_documents(reuters_test_X, reuters_test_y, 'reuters_test')
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
            test_size=0.2,
            shuffle=True,
            random_state=42,
        )
        reuters_train = Dataset.from_documents(reuters_train_X, reuters_train_y, 'reuters_train')
        reuters_train.to_file()
        reuters_validation = Dataset.from_documents(reuters_validation_X, reuters_validation_y, 'reuters_validation')
        reuters_validation.to_file()
        del reuters_train_and_validation_X, reuters_train_and_validation_y
        del reuters_train_X, reuters_train_y
        del reuters_validation_X, reuters_validation_y

    return reuters_train, reuters_validation, reuters_test


def ohsumed_read_file_worker(args):
    category_number, filename = args
    local_ohsumed_X = []
    local_ohsumed_y = []
    with open(filename, 'rt') as f:
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
        subprocess.call('make OHSUMED', shell=True)
        categories = chain(
            *(
                zip(
                    repeat(category_number),
                    glob('OHSUMED/ohsumed-all/{}/*'.format(category_name))
                )
                for category_number, category_name in (
                    (category_number, 'C{:02}'.format(category_number))
                    for category_number in range(1, 24)
                )
            )
        )
        ohsumed_X = []
        ohsumed_y = []
        with Pool(None) as pool:
            for local_ohsumed_X, local_ohsumed_y in pool.map(ohsumed_read_file_worker, categories):
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
            test_size=0.2,
            shuffle=True,
            random_state=42,
        )
        ohsumed_test = Dataset.from_documents(ohsumed_test_X, ohsumed_test_y, 'ohsumed_test')
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
            test_size=0.2,
            shuffle=False,
        )
        ohsumed_train = Dataset.from_documents(ohsumed_train_X, ohsumed_train_y, 'ohsumed_train')
        ohsumed_train.to_file()
        ohsumed_validation = Dataset.from_documents(ohsumed_validation_X, ohsumed_validation_y, 'ohsumed_validation')
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
        subprocess.call('make BBC', shell=True)
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
            test_size=0.2,
            shuffle=True,
            random_state=42,
        )
        bbcsport_test = Dataset.from_documents(
            bbcsport_test_X,
            bbcsport_test_y,
            'bbcsport_test',
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
            shuffle=False,
        )
        bbcsport_train = Dataset.from_documents(
            bbcsport_train_X,
            bbcsport_train_y,
            'bbcsport_train',
        )
        bbcsport_train.to_file()
        bbcsport_validation = Dataset.from_documents(
            bbcsport_validation_X,
            bbcsport_validation_y,
            'bbcsport_validation',
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
        subprocess.call('make BBC', shell=True)
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
            test_size=0.2,
            shuffle=True,
            random_state=42,
        )
        bbc_test = Dataset.from_documents(bbc_test_X, bbc_test_y, 'bbc_test')
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
            test_size=0.2,
            shuffle=False,
        )
        bbc_train = Dataset.from_documents(bbc_train_X, bbc_train_y, 'bbc_train')
        bbc_train.to_file()
        bbc_validation = Dataset.from_documents(bbc_validation_X, bbc_validation_y, 'bbc_validation')
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
        subprocess.call('make AMAZON', shell=True)
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
            train_size=0.8 / 100,
            test_size=0.2 / 100,
            shuffle=True,
            random_state=42,
        )
        amazon_test = Dataset.from_documents(amazon_test_X, amazon_test_y, 'amazon_test')
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
            test_size=0.2,
            shuffle=False,
        )
        amazon_train = Dataset.from_documents(amazon_train_X, amazon_train_y, 'amazon_train')
        amazon_train.to_file()
        amazon_validation = Dataset.from_documents(amazon_validation_X, amazon_validation_y, 'amazon_validation')
        amazon_validation.to_file()
        del amazon_train_and_validation_X, amazon_train_and_validation_y
        del amazon_train_X, amazon_train_y
        del amazon_validation_X, amazon_validation_y

    return amazon_train, amazon_validation, amazon_test


def load_newsgroups():
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
        (
            newsgroups_train_X,
            newsgroups_validation_X,
            newsgroups_train_y,
            newsgroups_validation_y
        ) = train_test_split(
            newsgroups_train_and_validation_raw.data,
            newsgroups_train_and_validation_raw.target,
            test_size=0.2,
            shuffle=True,
            random_state=42,
        )
        newsgroups_train = Dataset.from_documents(
            newsgroups_train_X,
            newsgroups_train_y,
            'newsgroups_train',
        )
        newsgroups_train.to_file()
        newsgroups_validation = Dataset.from_documents(
            newsgroups_validation_X,
            newsgroups_validation_y,
            'newsgroups_validation',
        )
        newsgroups_validation.to_file()
        del newsgroups_train_and_validation_raw
        del newsgroups_train_X, newsgroups_train_y
        del newsgroups_validation_X, newsgroups_validation_y

        newsgroups_test_raw = fetch_20newsgroups(subset='test')
        newsgroups_test_X = newsgroups_test_raw.data
        newsgroups_test_y = newsgroups_test_raw.target
        newsgroups_test = Dataset.from_documents(
            newsgroups_test_X,
            newsgroups_test_y,
            'newsgroups_test',
        )
        newsgroups_test.to_file()
        del newsgroups_test_raw
        del newsgroups_test_X, newsgroups_test_y

    return newsgroups_train, newsgroups_validation, newsgroups_test


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


def cached_sparsesvd(basename, *args):
    """Produces an SVD of a document matrix, loading it if cached.

    Parameters
    ----------
    basename : str
        The basename of the cached SVD matrix.
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

    filename = 'matrices/svd-{}.pkl.xz'.format(basename)
    try:
        with lzma.open(filename, 'rb') as f:
            LOGGER.debug('Loading SVD matrices {}.'.format(filename))
            ut, s, vt = pickle.load(f)
    except IOError:
        ut, s, vt = sparsesvd(*args)
        with lzma.open(filename, 'wb', preset=0) as f:
            LOGGER.info('Saving SVD matrices {}.'.format(filename))
            pickle.dump((ut, s, vt), f, pickle.HIGHEST_PROTOCOL)
    return (ut, s, vt)


def cached_sparse_term_similarity_matrix(basename, *args, **kwargs):
    """Produces a sparse term similarity matrix, loading it if cached.

    Parameters
    ----------
    basename : str
        The basename of the cached SVD matrix.
    args : iterable
        The arguments of the `SparseTermSimilarityMatrix` constructor.
    kwargs : dict
        The keyword arguments of the `SparseTermSimilarityMatrix` constructor.

    Returns
    -------
    term_matrix : gensim.similarities.SparseTermSimilarityMatrix
        The sparse term similarity matrix.
    """

    filename = 'matrices/termsim-{}.pkl.xz'.format(basename)
    try:
        with lzma.open(filename, 'rb') as f:
            LOGGER.debug('Loading term similarity matrix {}.'.format(filename))
            term_matrix = pickle.load(f)
    except IOError:
        term_sims = SparseTermSimilarityMatrix(*args, **kwargs)
        term_matrix = term_sims.matrix
        with lzma.open(filename, 'wb', preset=0) as f:
            LOGGER.info('Saving term similarity matrix {}.'.format(filename))
            pickle.dump(term_matrix, f, pickle.HIGHEST_PROTOCOL)
    return term_matrix


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
    document : list of list of (int, float)
        A document.

    Returns
    -------
    binarized_document : list of list of (int, float)
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
    bm25 : gensim.summarization.bm25.BM25
        Statistics about the documents in a dataset.
    doc_len : int
        The length of the document.

    Returns
    -------
    pivoted_document : list of list of (int, float)
        The pivoted document.
    """

    document, slope, bm25, doc_len = args
    pivoted_document = [
        (
            term_id,
            term_weight / (
                1 - slope + slope * (doc_len / bm25.avgdl)
            ),
        )
        for term_id, term_weight in document
    ]
    return pivoted_document


def translate_worker(args):
    """Translates a document from a source dictionary to a target dictionary.

    Parameters
    ----------
    document : list of list of (int, float)
        A document.
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
            pickle.dump(vars(self), f, pickle.HIGHEST_PROTOCOL)

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
    dictionary : gensim.corpora.Dictionary
        A mapping between tokens and token ids.
    target : {list, None}
        The document classes. Defaults to `None`.
    """
    def __init__(self, name, bm25, corpus, dictionary, target=None):
        self.name = name
        self.bm25 = bm25
        self.corpus = corpus
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
        with Pool(None) as pool:
            corpus = pool.map(tokenize_worker, documents)
        bm25 = BM25(corpus)
        dictionary = Dictionary(corpus, prune_at=None)
        if target is not None:
            target = list(target)
        dataset = Dataset(name, bm25, corpus, dictionary, target)
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
            dataset = Dataset(name, **pickle.load(f))
        return dataset

    def to_file(self, basename):
        """Saves a dataset to a file.

        """
        name = self.name
        filename = 'corpora/{}.pkl.xz'.format(name)
        with lzma.open(filename, 'wb', preset=0) as f:
            LOGGER.info('Saving dataset to file {}.'.format(filename))
            pickle.dump(vars(self), f, pickle.HIGHEST_PROTOCOL)

    def classify(self, validation, test, space='vsm', weights='bow', measure='inner_product', num_bits=32):
        """Performs classification using this dataset as the training set.

        Parameters
        ----------
        validation : Dataset
            The validation set.
        test : Dataset
            The test set.
        space : {'vsm', 'sparse_soft_vsm', 'dense_soft_vsm', 'lsi', 'lda'}, optional
            The document representation used for the classification.
        weights : {'binary', 'bow', 'tfidf', 'bm25'}, optional
            The term weighting scheme used for the classification.
        measure : {'inner_product', 'wmd', 'ntlm'}, optional
            The similarity measure used for the classification.
        num_bits : {1, 32}, optional
            The number of bits used to construct Word2Bit embeddings.
        """
        params = {'space': space, 'weights': weights, 'measure': measure, 'num_bits': num_bits}
        grid_specification = {}
        train = self

        LOGGER.debug('Preprocessing the datasets')
        if space in ('vsm', 'lsi', 'sparse_soft_vsm'):
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

        LOGGER.debug('Performing a grid search')
        for grid_params in grid_search(grid_specification):
            params.update(grid_params)
            if space in ('vsm', 'sparse_soft_vsm', 'lsi'):
                if measure == 'inner_product':
                    doc_sims = train.inner_product(validation, 'classification', params)
            results = []
            for k in range(1, 20):
                params['k'] = k
                result = ClassificationResult.from_similarities(doc_sims, train, validation, params)
                results.append(result)
        best_result = max(results)

        LOGGER.debug('Testing the performance')
        params = best_result.params
        if space in ('vsm', 'sparse_soft_vsm', 'lsi'):
            if measure == 'inner_product':
                doc_sims = train.inner_product(test, 'classification', params)
        result = ClassificationResult.from_similarities(doc_sims, train, test, params)
        return result

    def inner_product(self, queries, task, params):
        """Computes the inner product between two datasets.

        Parameters
        ----------
        queries : Dataset
            A dataset of queries.
        task : {'classification', 'adhoc_ir'}
            The intent of the inner product (text classification or ad-hoc
            information retrieval), which affects the tf-idf weighting scheme
            (dtb.dtb for text classification and dtb.nnn for ad-hoc information
            retrieval).
        params : dict
            The parameters of the vector space model.

        Returns
        -------
        inner_product : np.matrix
            The inner product between the two datasets.
        """

        weights = params['weights']
        space = params['space']
        if weights == 'tfidf':
            slope = params['slope']
        elif weights == 'bm25':
            k1 = params['k1']

        collection = self
        collection_bm25 = collection.bm25
        if weights == 'tfidf':
            collection_corpus = map(collection.dictionary.doc2bow, collection.corpus)
            collection_tfidf = TfidfModel(dictionary=collection.dictionary, smartirs='dtn')
            collection_corpus = map(pivot_worker, zip(
                collection_tfidf[collection_corpus],
                repeat(slope),
                repeat(collection_bm25),
                collection_bm25.doc_len,
            ))
            collection_corpus = map(translate_worker, zip(
                collection_corpus,
                repeat(collection.dictionary),
                repeat(common_corpus.dictionary),
            ))
        else:
            collection_corpus = map(common_corpus.dictionary.doc2bow, collection.corpus)
            if weights == 'bow':
                collection_corpus = map(unitvec, collection_corpus)
            elif weights == 'binary':
                collection_corpus = map(binarize_worker, collection_corpus)
            elif weights == 'bm25':
                collection_corpus = map(bm25_worker, zip(
                    collection_corpus,
                    repeat(k1),
                    repeat(0.25),
                    repeat(collection_bm25),
                    collection_bm25.doc_len,
                    repeat(common_corpus.dictionary),
                ))
        collection_corpus = list(collection_corpus)
        collection_matrix = corpus2csc(collection_corpus, len(common_corpus.dictionary))

        if weights == 'tfidf' and task == 'classification':
            query_corpus = map(collection.dictionary.doc2bow, queries.corpus)
            query_corpus = map(pivot_worker, zip(
                collection_tfidf[query_corpus],
                repeat(slope),
                repeat(collection_bm25),
                collection_bm25.doc_len,
            ))
            query_corpus = map(translate_worker, zip(
                query_corpus,
                repeat(collection.dictionary),
                repeat(common_corpus.dictionary),
            ))
        else:
            query_corpus = map(common_corpus.dictionary.doc2bow, queries.corpus)
            if weights in ('binary', 'bm25', 'tfidf'):
                query_corpus = map(binarize_worker, query_corpus)
            elif weights == 'bow':
                query_corpus = map(unitvec, query_corpus)
        query_corpus = list(query_corpus)
        query_matrix = corpus2csc(query_corpus, len(common_corpus.dictionary))

        if space == 'vsm':
            doc_sims = collection_matrix.T.dot(query_matrix).T.todense()
        elif space == 'lsi':
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
            ut, s, vt = cached_sparsesvd(lsi_basename, collection_matrix, 500)
            collection_matrix = vt
            query_matrix = np.diag(1 / s).dot(ut.dot(query_matrix.todense()))
            del ut
            collection_matrix_norm = np.multiply(collection_matrix.T, collection_matrix.T).sum(axis=1).T
            query_matrix_norm = np.multiply(query_matrix.T, query_matrix.T).sum(axis=1).T
            collection_matrix = np.multiply(collection_matrix, 1 / np.sqrt(collection_matrix_norm))
            query_matrix = np.multiply(query_matrix, 1 / np.sqrt(query_matrix_norm))
            collection_matrix[collection_matrix == np.inf] = 0.0
            query_matrix[query_matrix == np.inf] = 0.0
            doc_sims = collection_matrix.T.dot(query_matrix).T
        elif space == 'sparse_soft_vsm':
            num_bits = params['num_bits']
            tfidf = params['tfidf']
            symmetric = params['symmetric']
            positive_definite = params['positive_definite']
            nonzero_limit = params['nonzero_limit']
            term_basename = '{num_bits}-{tfidf}-{symmetric}-{positive_definite}-{nonzero_limit}'.format(
                num_bits=num_bits,
                tfidf=tfidf,
                symmetric=symmetric,
                positive_definite=positive_definite,
                nonzero_limit=nonzero_limit,
            )
            term_index = WordEmbeddingSimilarityIndex(common_vectors[num_bits])
            term_matrix = cached_sparse_term_similarity_matrix(
                term_basename,
                term_index,
                common_corpus.dictionary,
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


# try:
#     common_corpus = Dataset.from_file('fil9')
# except IOError:
#     subprocess.call('make corpora', shell=True)
#     with open('corpora/fil9', 'rt') as f:
#         common_corpus = Dataset.from_documents(f, 'fil9')
#         common_corpus.to_file()
# common_tfidf = TfidfModel(dictionary=common_corpus.dictionary, smartirs='dtn')

# subprocess.call('make vectors', shell=True)
common_vectors = {
    # 1: KeyedVectors.load_word2vec_format('vectors/1b_1000d_vectors_e50_nonbin', binary=False),
    # 32: KeyedVectors.load_word2vec_format('vectors/32b_1000d_vectors_e50_nonbin', binary=False),
}
