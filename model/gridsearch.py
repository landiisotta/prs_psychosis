from utils import scorers_binary, model_dict, rcv_splits, read_data
from sklearn.model_selection import GridSearchCV
from logger.logger import setup_logging, read_json
from multiprocessing import Pool
import logging
import argparse
import time
import sys
import pandas as pd
import numpy as np
from imblearn.over_sampling import RandomOverSampler


def gridsearch(model,
               data,
               label,
               cv,
               param_grid,
               score_best,
               train_score=True):
    """
    Run grid search repeated CV. Return best estimator when only one evaluation metric is selected.

    :param model: Model
    :type model: class object
    :param data: Dataset
    :type data: numpy array (n_samples, n_features)
    :param cv: Cross validation indices
    :type cv: numpy array
    :param param_grid: Parameter grids
    :type param_grid: dict
    :param label: Labels
    :type label: numpy array (n_samples,)
    :param score_best: Evaluate grid-search performance on this metric, default 'AUPRC'
    :type score_best: str, if None all scorers are evaluated and best model is not returned
    :param train_score: Whether to return train score as well along with validation, default True
    :type train_score: bool
    :return: grid-search object, with fitted model if `score_best is None`
    :rtype: grid-search object with attributes.
    """
    if isinstance(score_best, str):
        refit_score = score_best
    else:
        refit_score = False
    scoring = scorers_binary
    grid = GridSearchCV(model,
                        param_grid=param_grid,
                        cv=cv,
                        scoring=scoring,
                        refit=refit_score,
                        return_train_score=train_score)

    grid.fit(data, label)
    return grid


def run_gridsearch(data, label, gs_params, rcv_params, score_best, oversampling=False, save_out=None,
                   covariate=None):
    """
    Function that runs grid search in parallel with repeated cross validation as defined in
    the configuration file. It returns the grid-search output list. If `save_out` is True it
    writed the cross-validation dataframe to file.

    :param data: Dataset
    :type data: numpy array (n_samples, n_features)
    :param label: Labels
    :type label: numpy array (n_samples,)
    :param gs_params: Grid-search parameters
    :type gs_params: dict
    :param rcv_params: Repeated cross-validation parameters
    :type rcv_params: dict
    :param oversampling: Enable oversampling in case of imbalanced classes
    :type: bool
    :param save_out: Whether to write the CV results to file, default False
    :type save_out: bool
    :param covariate: ancestry covariate for stratified cv
    :type: str
    :return: parallel run grid-search output
    :rtype: list
    """
    cv_idx = rcv_splits(data, label,
                        rcv_params['repeat'],
                        rcv_params['folds'],
                        cov=covariate)

    if oversampling:
        if len(np.unique(label)) == 2:
            cl, count = np.unique(label, return_counts=True)
            perc = [c / max(count) for c in count]
            sampstrat = min(perc) + 0.2
            if sampstrat > 1.0:
                sampstrat = 1
            over = RandomOverSampler(sampling_strategy=sampstrat, random_state=42)
        else:
            over = RandomOverSampler(sampling_strategy='not majority', random_state=42)
        cv_idx_rsmpl = []
        for idx_train, idx_val in cv_idx:
            cv_idx_rsmpl.append((over.fit_resample(idx_train.reshape(-1, 1), label[idx_train])[0].reshape(1, -1)[0],
                                 over.fit_resample(idx_val.reshape(-1, 1), label[idx_val])[0].reshape(1, -1)[0]))
        cv_idx = cv_idx_rsmpl

    pool = Pool(processes=len(gs_params))
    param_it = [(model_dict[mname],
                 data, label, cv_idx, param_dict, score_best)
                for mname, param_dict in gs_params.items()]
    grid_out = pool.starmap(gridsearch, param_it)
    pool.terminate()
    if save_out is not None:
        for mname, g in zip(gs_params.keys(), grid_out):
            pd.DataFrame(g.cv_results_).to_csv(f'./{save_out}/gridsearch_{mname}scores.txt')
            # joblib.dump(g.best_estimator_, f'./{save_out}/gridsearch_{mname}bestestimator.pkl')
            # logging.info(f'Model {mname} best result:')
            # logging.info(f'Best params: {g.best_params_}')
            # logging.info(f'Best score: {g.best_score_}')
            logging.info('\n')
    return grid_out


def main(configure):
    """
    Main function to perform grid-search.

    :param configure: Configuration file with parameters to try for each baseline model
    """
    config_params = read_json(configure.config_file)
    data_tr, label_tr = read_data(configure.training_set,
                                  configure.cat_feat)
    data_tr = np.array(data_tr)
    if configure.covariate is not None:
        run_gridsearch(data_tr[:, :-1],
                       label_tr,
                       config_params['run_gridsearch'],
                       config_params['rcv'],
                       oversampling=configure.oversampling,
                       score_best=configure.scorer,
                       save_out=configure.output_folder,
                       covariate=configure.covariate)
    else:
        run_gridsearch(data_tr[:, :-1],
                       label_tr,
                       config_params['run_gridsearch'],
                       config_params['rcv'],
                       oversampling=configure.oversampling,
                       score_best=configure.scorer,
                       save_out=configure.output_folder,
                       covariate=list(data_tr[:, -1]))


if __name__ == '__main__':
    setup_logging(save_dir='../out',
                  name_file='gridsearch.log')

    parser = argparse.ArgumentParser(description='Run model (grid search)')
    parser.add_argument('-c', '--config', type=str, dest='config_file',
                        help='Add configuration file path')
    parser.add_argument('-tr', '--training_set',
                        type=str, dest='training_set',
                        default=None,
                        help='Include training set')
    parser.add_argument('-s',
                        '--scorer', type=str,
                        dest='scorer', default=None,
                        help='Performance metric')
    parser.add_argument('-cat',
                        '--cat_feat', type=str,
                        dest='cat_feat', default=None,
                        help='Indices corresponding to categorical variables')
    parser.add_argument('-cov',
                        '--covariate', type=str,
                        dest='covariate', default=None,
                        help='Covariate for stratification of cv folds')
    parser.add_argument('-o',
                        '--out', type=str,
                        dest='output_folder', default=None,
                        help='Output folder')
    parser.add_argument('-ovs',
                        '--oversampling', type=str,
                        dest='oversampling', default=False,
                        help='Enable oversampling')
    config = parser.parse_args(sys.argv[1:])
    logging.info('')

    start = time.time()
    main(configure=config)
    logging.info('\nProcessing time: %s seconds\n' % round(time.time() - start, 2))

    logging.info('Task completed\n')
