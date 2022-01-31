import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
    SGDClassifier
)

from sklearn.cluster import KMeans

from sklearn.svm import (
    LinearSVC,
    SVC
)

from sklearn.metrics import (
    precision_recall_curve, 
    f1_score, 
    roc_curve,
    auc,
    precision_score,
    recall_score,
    mean_squared_error,
    confusion_matrix
)

from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

import collections
import re

# from scipy.stats import pearsonr, interval
import scipy.stats as st

def plot_countbar(data: pd.DataFrame, product: str, col1: str):
    """
    given a product, return and save a countplot with column "col1" in data
    """
    plt.figure(figsize=(10,8))
    sns.countplot(x = data[col1], order = data[col1].value_counts().index)
    plt.xticks(rotation=90)
    plt.xlabel(f'Reviewers by {col1}')
    plt.savefig(f'plots/{product}_reviewers_by_{col1}.jpeg')


def plot_diff_in_means(data: pd.DataFrame, product: str, col1: str, col2: str = 'rating'):
    """
    given a product, plots difference in means across groups and saves the figure
    col1
        categorical data with groups
    col2
        continuous data for the means
    """
    
    data_groupby = data.groupby(col1, as_index=False)[col2]
    data_groupby_labels = data_groupby.agg('mean').sort_values(col2, ascending = False)

    data_agg = pd.DataFrame(data_groupby_labels).rename(columns = {col2: 'mean'})

    n = data_groupby.count()
    data_agg = data_agg.merge(n, how = 'left', right_on=col1, left_on = col1).rename(columns = {col2: 'n'})

    std = data_groupby.agg(np.std)
    data_agg = data_agg.merge(std, how = 'left', right_on=col1, left_on = col1).rename(columns = {col2: 'std'})
    data_agg['se'] = data_agg['std']/  np.sqrt(data_agg['n'])
    
    data_agg['lower'] = st.t.interval(alpha = 0.95, df =data_agg['n']-1, loc = data_agg['mean'], scale = data_agg['se'])[0]
    data_agg['upper'] = st.t.interval(alpha = 0.95, df =data_agg['n']-1, loc = data_agg['mean'], scale = data_agg['se'])[1]
    

    for upper, mean, lower, y in zip(data_agg['upper'], data_agg['mean'], data_agg['lower'], data_agg[col1]):
        plt.plot((lower, mean, upper), (y, y, y), 'b.-')
    plt.yticks(range(len(n)), list(data_agg[col1]))
    plt.savefig(fname = f'plots/{product}_{col1}_diff_in_means.jpeg')


def plot_cross_tab_heatmap(data: pd.DataFrame, product: str, col1: str, col2: str, col3: str, func: str = 'mean'):
    """
    given a product, cross-tabulate col1 and col2 in data, and create a heatmap given func with col3 as the color bar
    col1
        categorical data in x axis
    col2
        categorical data in y axis
    col3
        continuous data in color bar
    func
        'mean' or 'median' for aggfunc in pd.crosstab
    """
    col1_col2_crosstab = pd.crosstab(data[col1], data[col2], values = data[col3], aggfunc = func)
    col1_labels = data.groupby(col1, as_index = False).count()[col1]
    col2_labels = data.groupby(col2, as_index = False).count()[col2]

    fig, ax = plt.subplots(figsize=(10,8))
    im = ax.imshow(col1_col2_crosstab)
    ax.set_yticks(np.arange(len(col1_labels)), labels=col1_labels)
    ax.set_xticks(np.arange(len(col2_labels)), labels=col2_labels)
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(col3, rotation=-90, va="bottom")

    kw = dict(horizontalalignment="center",
                verticalalignment="center")
    # threshold = im.norm(col1_col2_crosstab.max())/2
    texts = []
    textcolors=("black", "white")
    for i in range(col1_col2_crosstab.shape[0]):
        for j in range(col1_col2_crosstab.shape[1]):
            kw.update(color=textcolors[int(im.norm(col1_col2_crosstab.iloc[i, j]) < 0.5)])
            text = im.axes.text(j, i, round(col1_col2_crosstab.iloc[i, j], 2), kw)
            texts.append(text)
    plt.title(product.replace('_', ' '))
    plt.savefig(fname = f'plots/{product}_{col1}_{col2}_crosstab_heatmap.jpeg')
    



def plot_auc_roc(predict_y, val_y, filename: str = "auc_roc.jpeg"):

    thresholds = np.linspace(0, 1, 10)

    fpr = []
    tpr = []

    for threshold in thresholds:
        predict_y_binary = np.where(predict_y >= threshold, 1, 0)
            
        fp = np.sum((val_y == 0) & (predict_y_binary == 1)) # true value is 0 but predict to be 1
        tp = np.sum((val_y == 1) & (predict_y_binary == 1)) # true value is 1 & predict to be 1

        fn = np.sum((val_y == 1) & (predict_y_binary == 0)) # true value is 1 but predict to be 0
        tn = np.sum((val_y == 0) & (predict_y_binary == 0)) # true value is 0 but predict to be 0

        fpr.append( fp / (fp + tn))
        tpr.append( tp / (tp + fn))

        ## ROC & ROC-AUC
    roc_auc_ = round(auc(fpr, tpr), 5)
    fig, ax3 = plt.subplots(1, 1)
    ax3.plot(fpr, tpr, label = "ROC")
    ax3.plot([0, 1], [0, 1], 'k--')
    ax3.set_xlabel("False Positive Rate")
    ax3.set_ylabel("True Positive Rate")
    ax3.text(0.55,0.2, 'AUC = {}'.format(roc_auc_))
    plt.savefig(f'plots/{filename}')


def plot_recision_recall_f1(predict_y, val_y, filename: str = "precision_recall_f1.jpeg"):

    thresholds = np.linspace(0, 1, 10)
    
    precisions = []
    recalls = []
    f1_scores = []

    for threshold in thresholds:
        predict_y_binary = np.where(predict_y >= threshold, 1, 0)
    
        precisions.append(precision_score(val_y, predict_y_binary))
        recalls.append(recall_score(val_y, predict_y_binary))
        f1_scores.append(f1_score(val_y, predict_y_binary))
    
    precision_at_threshold = precision_score(val_y, np.where(predict_y >= 0.5, 1,0))
    recall_at_threshold = recall_score(val_y, np.where(predict_y >= 0.5, 1,0))
    f1_score_at_threshold = f1_score(val_y, np.where(predict_y >= 0.5, 1,0))
        
    ### plot precisions, recalls, and F1 scores across threshold
    fig, ax2 = plt.subplots(1, 1)
    ax2.plot(thresholds, precisions, label = "precision")
    ax2.plot(thresholds, recalls, label = "recall")
    ax2.plot(thresholds, f1_scores, label = "f1 score")
    ax2.axvline(0.5, linestyle = ':', color = 'r') 
    ax2.plot(0.5, precision_at_threshold, "s")
    ax2.annotate("{}".format(round(precision_at_threshold,3)), (0.5, precision_at_threshold))
    ax2.plot(0.5, recall_at_threshold, "s")
    ax2.annotate("{}".format(round(recall_at_threshold,3)), (0.5, recall_at_threshold))
    ax2.plot(0.5, f1_score_at_threshold, "s")
    ax2.annotate("{}".format(round(f1_score_at_threshold,3)), (0.5, f1_score_at_threshold))
    ax2.legend()
    ax2.set_xlabel("Thresholds")
    plt.savefig(f'plots/{filename}')