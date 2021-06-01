import matplotlib.pyplot as plt
import re
import joblib
import numpy as np
from sklearn.preprocessing import label_binarize


def plot_gridmetrics(metrics,
                     scorers,
                     color_list,
                     ylim_v,
                     yticks_v,
                     setoff=0.012,
                     rot=0,
                     figsize=(5, 5),
                     horizontalalignment="left",
                     ylab="AUPRC",
                     file_name=None,
                     train_score=False):
    """
    Plot selected metrics for grid search parameter combinations.
    Mark the best parameter combination for each metric.

    :param metrics: dataframe with grid search attribute cv_results_
    :type metrics: pandas dataframe
    :param scorers: metric to evaluate the best solution
    :type scorers: list of str
    :param color_list: List of colors to use for different metrics
    :type color_list: list of str
    :pram rot: label rotation to improve readability, default 0
    :type rot: int
    :param file_name: If not None then save plot to pdf, default None
    :type file_name: str
    :param ylab: metric used
    :type ylab: str
    :param figsize: figure size, default (5, 5)
    :type figsize: tuple
    :param train_score: whether to report performance scores on training set
    :type train_score: bool
    """
    plt.figure(figsize=figsize)

    plt.xlabel("C", fontsize=18)

    ax = plt.gca()

    names = [k for k in metrics.keys() if re.match('param_', k)]
    x_axis = np.array(metrics[names[0]], dtype=str)
    for scorer, color in zip(sorted(scorers), color_list):
        if train_score:
            lines = (('train', '--'), ('test', '-'))
            for sample, style in lines:
                _plot_lines(ax, metrics, x_axis, sample, scorer, style, color)
        else:
            _plot_lines(ax, metrics, x_axis, 'test', scorer, '-', color)

        best_index = np.nonzero(np.array(metrics['rank_test_%s' % scorer] == 1))[0][0]
        best_score = metrics['mean_test_%s' % scorer][best_index]
        std_score = metrics['std_test_%s' % scorer][best_index]

        # Plot a dotted vertical line at the best score for that scorer marked by x
        ax.plot(x_axis[best_index], best_score,
                linestyle='-.', color=color, marker='.', markeredgewidth=3, ms=8)

        # Annotate the best score for that scorer
        ax.annotate("%0.3f (%0.3f)" % (best_score, std_score),
                    (x_axis[best_index], best_score - setoff),
                    horizontalalignment=horizontalalignment, fontsize=12)
    plt.xticks(fontsize=14, rotation=rot)
    plt.ylim(ylim_v[0], ylim_v[1])
    plt.yticks(yticks_v, fontsize=14)
    plt.ylabel(ylab, fontsize=18)

    plt.legend(loc="best", labels=('train', 'val'), fontsize=12)
    plt.grid(False)
    if file_name is not None:
        plt.savefig(fname=file_name, format='pdf')
    else:
        plt.show()
    return


def plot_auprc(out_folder, metric, save):
    """
    Plot AUPRC curves for test sets
    """
    label, _, _, _, prcl, rccl = joblib.load(f'../{out_folder}/clinical/{metric}_clinical.pkl')
    _, _, _, _, prgen, rcgen = joblib.load(f'../{out_folder}/genetic/{metric}_genetic.pkl')
    _, _, _, _, prall, rcall = joblib.load(f'../{out_folder}/all/{metric}_all.pkl')

    plt.figure(figsize=(6, 6))
    plt.plot(rccl, prcl, linestyle='-', label='Clinical')
    plt.plot(rcgen, prgen, linestyle='-', label='PCs + PRS')
    plt.plot(rcall, prall, linestyle='-', label='All features')
    if len(np.unique(label)) > 2:
        y_label = label_binarize(label, classes=np.unique(label))
        plt.hlines(sum(y_label.ravel()) / len(y_label.ravel()), xmin=0, xmax=1, colors='black', linestyles='-',
                   label='Random classifier')
    else:
        plt.hlines(sum(label) / len(label), colors='black', linestyles='-', xmin=0, xmax=1,
                   label='Random classifier')
    plt.xlim(0, 1)
    plt.xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=14)
    plt.ylim(0, 1)
    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=14)
    # axis labels
    plt.xlabel('Recall', fontsize=18)
    plt.ylabel('Precision', fontsize=18)
    # show the legend
    plt.legend(fontsize=12)
    plt.savefig(fname=f'../{out_folder}/{save}', format='pdf')
    # show the plot
    plt.show()


def plot_auprc_binarized(out_folder, metric, save):
    """
    Plot AUPRC curves with binarized PRS.
    """
    label, _, _, _, prgen, rcgen = joblib.load(f'../{out_folder}/genetic_binarized/{metric}_genetic.pkl')
    _, _, _, _, prall, rcall = joblib.load(f'../{out_folder}/all_binarized/{metric}_all.pkl')

    plt.figure(figsize=(6, 6))
    plt.plot(rcgen, prgen, linestyle='-', label='PCs + PRS')
    plt.plot(rcall, prall, linestyle='-', label='All features')
    if len(np.unique(label)) > 2:
        y_label = label_binarize(label, classes=np.unique(label))
        plt.hlines(sum(y_label.ravel()) / len(y_label.ravel()), xmin=0, xmax=1, colors='black', linestyles='-',
                   label='Random classifier')
    else:
        plt.hlines(sum(label) / len(label), xmin=0, xmax=1, colors='black', linestyles='-',
                   label='Random classifier')
    plt.xlim(0, 1)
    plt.xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=14)
    plt.ylim(0, 1)
    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=14)
    # axis labels
    plt.xlabel('Recall', fontsize=18)
    plt.ylabel('Precision', fontsize=18)
    # show the legend
    plt.legend(fontsize=12)
    plt.savefig(fname=f'../{out_folder}/{save}', format='pdf')
    # show the plot
    plt.show()


"""
Private functions
"""


def _plot_lines(ax, metrics, x_axis, sample, scorer, style, color):
    """
    Plot metric lines.
    :param ax:
    :param metrics:
    :param x_axis:
    :param sample:
    :param scorer:
    :param style:
    :param color:
    :return:
    """
    sample_score_mean = metrics['mean_%s_%s' % (sample, scorer)]
    sample_score_std = metrics['std_%s_%s' % (sample, scorer)]
    ax.errorbar(x_axis, sample_score_mean, yerr=sample_score_std, color='black',
                alpha=1 if sample == 'test' else 0)
    ax.plot(x_axis, sample_score_mean, style, color=color,
            alpha=1 if sample == 'test' else 0.7,
            label="%s (%s)" % (scorer, sample))
