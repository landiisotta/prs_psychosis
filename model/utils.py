import json
import logging
from pathlib import Path
from collections import OrderedDict
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, average_precision_score, precision_score, recall_score, fbeta_score
from sklearn.model_selection import RepeatedStratifiedKFold
import csv
import numpy as np
import pandas as pd

model_dict = {"LR": LogisticRegression(random_state=42),
              "RF": RandomForestClassifier(random_state=42),
              "SVM": SVC(random_state=42)}


def auprc(estimator, X, y):
    return average_precision_score(y, estimator.predict_proba(X)[:, 1])


scorers_binary = {
    'AUPRC': auprc,
    'precision': make_scorer(precision_score, zero_division=0),
    'recall': make_scorer(recall_score),
    'F2': make_scorer(fbeta_score, beta=2)}


class Memoize:
    """
    Memoization class
    """

    def __init__(self, fn):
        self.fn = fn
        self.memo = {}

    def __call__(self, *args):
        if args not in self.memo:
            self.memo[args] = self.fn(*args)
        return self.memo[args]


@Memoize
def read_data(dataset, cat_feat_file=None):
    """
    Function that takes as input a file with the
    dataset name and returns two arrays with data
    and labels. Categorical variables are one-hot encoded and
    appended at the end of the numpy array.

    :param dataset: Dataset file name
    :type dataset: str
    :param cat_feat_file: File that includes indices of the categorical variables
    :type cat_feat_file: str
    :return: ndarray (data, label)
    """
    if cat_feat_file is None:
        catfeat = []
    else:
        with open(cat_feat_file, 'r') as f:
            rd = csv.reader(f)
            next(rd)
            catfeat = [int(r[0]) for r in rd]
    with open(dataset, 'r') as f:
        rd = csv.reader(f, delimiter='\t')
        next(rd)
        data, label = [], []
        for r in rd:
            try:
                data.append([int(x) if idx in catfeat else float(x) for idx, x in enumerate(r[:-2])] + [r[-2]])
            except ValueError:
                data.append([float(x) for x in r[:-2]] + [r[-2]])
            label.append(int(r[-1]))
    data, label = np.array(data), np.array(label)
    return data, label


def rcv_splits(data, labels, repeats, folds, cov=None):
    """
    Function that splits data and labels into folds for repeated CV
    as specified by `repeats` and `folds` parameters and returns indices.

    :param data: Dataset
    :type data: numpy array (n_samples, n_features)
    :param labels: Dataset labels
    :type labels: numpy array (n_samples,)
    :param repeats: Number of repeated cross validation folds
    :type repeats: int
    :param folds: Number of cross-validation folds
    :type folds: int
    :return: list of indices for each fold
    :rtype: list
    """
    logging.info(f"Running {repeats}x{folds} "
                 f"cross-validation...")
    rcv_index = RepeatedStratifiedKFold(n_repeats=repeats,
                                        n_splits=folds,
                                        random_state=42)
    if isinstance(cov, str):
        covariate = pd.read_csv(cov)
        strat_vect = ['-'.join([str(lab), cova]) for lab, cova in zip(labels, covariate.ancestry)]
    else:
        strat_vect = labels
    idx_it = [(train_idx, val_idx) for train_idx, val_idx in rcv_index.split(data, strat_vect)]
    return idx_it


def read_json(fname):
    """
    Function that takes as input the json configuration file name
    and imports it as an ordered dictionary.
    :param fname: Name of configuration file
    :type fname: str
    :return: ordered dict of configuration parameters
    :rtype: OrderedDict
    """
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)
