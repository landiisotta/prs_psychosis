import argparse
import sys
import time
import logging
from utils import read_data, model_dict
from logger.logger import setup_logging
import joblib
from sklearn.metrics import average_precision_score, precision_recall_curve, precision_score, recall_score, fbeta_score
from sklearn.preprocessing import label_binarize
import numpy as np
import csv
import re
import pandas as pd
from logger.logger import read_json
import os


def evaluate(best_model, test, label, scoring):
    """
    Evaluate best model returned by grid-search

    :param best_model: Fitted model
    :type best_model: class object
    :param test: Test set
    :type test: numpy array (n_samples, n_features)
    :param label: Classes
    :type label: numpy array (n_samples,)
    :param scoring: Performance metrics
    :type: str
    :return: Labels, predicted labels, evaluation score
    :rtype: numpy array, numpy array, float
    """
    logging.info(f"Testing {best_model}...")
    if not re.search('multiclass', scoring):
        pred_prob = best_model.predict_proba(test)[:, 1]
        pred_lab = best_model.predict(test)
        score_out = average_precision_score(label, pred_prob)
        precision = precision_score(label, pred_lab, zero_division=0)
        recall = recall_score(label, pred_lab)
        f2 = fbeta_score(label, pred_lab, beta=2)
        logging.info(f"AUPRC on test set: {score_out}")
        logging.info(f"F2 on test set: {f2}")
        logging.info(f"Precision: {precision} -- Recall: {recall}")
        logging.info(f"Random prediction average precision: {sum(label) / len(label)}")
        precision, recall, _ = precision_recall_curve(label, pred_prob)
    else:
        y_test = label_binarize(label, classes=np.unique(label))
        pred_prob = best_model.predict_proba(test)
        pred_lab = best_model.predict(test)
        score_out = average_precision_score(y_test, pred_prob, average='micro')
        precision = precision_score(label, pred_lab, average="micro")
        recall = recall_score(label, pred_lab, average="micro")
        f2 = fbeta_score(label, pred_lab, beta=2, average='micro')
        logging.info(f"Multiclass AUPRC score (micro average): {score_out}")
        logging.info(f"Precision: {precision} -- Recall: {recall}")
        logging.info(f"F2 on test set: {f2}")
        logging.info(f"Random prediction average precison: {sum(y_test.ravel()) / len(y_test.ravel())}")
        precision, recall, _ = precision_recall_curve(y_test.ravel(), pred_prob.ravel())
    return label, pred_prob, score_out, pred_lab, precision, recall


def main(configure):
    """
    Run model main function.

    :param configure: Configuration file
    """
    config_params = read_json(configure.config_file)
    data_ts, label_ts = read_data(configure.test_set)
    data_ts = np.array(data_ts)
    logging.info(f"Test set: {configure.test_set}...")
    if configure.best_model is not None:
        best_model = joblib.load(configure.best_model)
    else:
        cv_results = pd.read_csv(configure.grid_search)
        data_tr, label_tr = read_data(configure.train_set)
        data_tr = np.array(data_tr)
        scores = cv_results[f'mean_test_{configure.scoring}']
        best_res = cv_results.loc[cv_results[f'mean_test_{configure.scoring}'] == max(scores)]
        best_params = eval(best_res.params.iloc[0])
        model = model_dict[config_params["models"]].set_params(**best_params)
        best_model = model.fit(data_tr[:, :-1], label_tr)
        joblib.dump(best_model, os.path.join(configure.output_path,
                                             'gridsearch_LRbestestimator_retrained.pkl'))
    label, pred_prob, score_out, pred_lab, precision, recall = evaluate(best_model, data_ts[:, :-1],
                                                                        label_ts,
                                                                        scoring=configure.scoring)
    joblib.dump((label, pred_lab, pred_prob, score_out, precision, recall),
                os.path.join(configure.output_path, "best_model_eval.pkl"))
    with open("../out/predicted_prob.txt", "w") as f:
        wr = csv.writer(f)
        wr.writerow(["label", "predicted", "pred_prob"])
        for lab, p, pp in zip(label, pred_lab, pred_prob):
            wr.writerow([lab, p, pp])


if __name__ == '__main__':
    setup_logging(save_dir='../out',
                  name_file='baseline_cv.log')

    parser = argparse.ArgumentParser(description='Evaluate model')
    parser.add_argument('-c', '--config', type=str, dest='config_file',
                        help='Add configuration file path')
    parser.add_argument('-bm', '--best_model',
                        type=str, dest='best_model',
                        default=None,
                        help='Best fitted model')
    parser.add_argument('-gs', '--grid_search',
                        type=str, dest='grid_search',
                        default=None,
                        help='Grid search output')
    parser.add_argument('-tr', '--training_set',
                        type=str, dest='training_set',
                        default=None,
                        help='Test set')
    parser.add_argument('-ts', '--test_set',
                        type=str, dest='test_set',
                        default=None,
                        help='Test set')
    parser.add_argument('-s', '--scoring',
                        type=str, dest='scoring',
                        default=None,
                        help='Scoring')
    parser.add_argument('-o',
                        '--out', type=str,
                        dest='output_path', default=None,
                        help='Output path')
    config = parser.parse_args(sys.argv[1:])
    logging.info('')

    start = time.time()
    main(configure=config)
    logging.info('\nProcessing time: %s seconds\n' % round(time.time() - start, 2))

    logging.info('Task completed\n')
