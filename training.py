import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder


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

import os

# from scipy.stats import pearsonr, interval
import scipy.stats as st

class feature_engineering():
    """
    loads data from pickle by product_name.pkl
    """

    def __init__(self, product_name: str) -> None:
        self.data = pd.read_pickle(f'data_full_review_cleaned/{product_name}')
        self.all_features = [
            'hair_color', 'eye_color', 'skin_tone', 'skin_type', 
            'skin_tone_cat', 'finish', 'coverage', 'shade_match', 
            'gifted', 'days_since_launch_scaled', 'month_of_purchase'
            ]
        self.one_hot_features = [
            'hair_color', 'eye_color', 'skin_tone', 'skin_type'
        ]
        self.other_features = [
            'skin_tone_cat', 'finish', 'coverage', 'shade_match', 
            'gifted', 'days_since_launch_scaled', 'month_of_purchase'
        ]
        self.X = self.data[self.all_features]
        self.y = self.data['recommended']
        # perhaps some logging to know if there's class imbalance issues?

    def train_test_split(self, proportion: float = 0.3):
        self.train_X, self.val_X, self.train_y, self.val_y = train_test_split(self.X, self.y, test_size=proportion, random_state=0)

    def one_hot_encoding(self, cols: list):
        enc_rest = OneHotEncoder(sparse=False)

        train_X_transform = enc_rest.fit_transform(self.train_X[cols])
        val_X_transform = enc_rest.transform(self.val_X[cols])

        train_X_transform = pd.DataFrame(train_X_transform)
        val_X_transform = pd.DataFrame(val_X_transform)

        return train_X_transform, val_X_transform


    def cross_one_hot_features(self, one_hot_col1: pd.DataFrame, one_hot_col2: pd.DataFrame, data: pd.DataFrame, col1: str = 'skin_type', col2: str = 'skin_tone'):
        """
        given one-hot encoded feature 1 (one_hot_col1) and one-hot encoded feature 2 (one_hot_col1),
        returns a dataframe with crossed feature b/w one_hot_col1 and one_hot_col2
        columns of the returned dataframe are named by each of the crossed categories in col1 and col2 as "col1_col2"
        """
        total_col1_cat = len(data.groupby([col1],as_index=False).count()) - 1
        total_col2_cat = len(data.groupby([col2],as_index=False).count()) - 1

        data_cross = pd.DataFrame()
        
        i = 0 # col1
        
        while i <= total_col1_cat:
            j = 0 # col2
            while j <= total_col2_cat:
                new_cross = one_hot_col1[i] * one_hot_col2[j]
                new_cross = pd.Series(new_cross)
                col1_name = data.groupby([col1],as_index=False).count()[col1][i]
                col2_name = data.groupby([col2],as_index=False).count()[col2][j]
                data_cross[f'{col1_name}_{col2_name}'] = new_cross
                j += 1
            i += 1
            
        return data_cross
    
    def concat_all_features(
            self, 
            train_X_transformed: pd.DataFrame, 
            val_X_transformed: pd.DataFrame
        ):
        """
        concatenates non-one-hot-encoded features with one-hot-encoded ones
        """
        for i in self.other_features:
            train_X_transformed[i] = self.train_X[i].reset_index(drop = True)
            val_X_transformed[i] = self.val_X[i].reset_index(drop = True)
        return train_X_transformed, val_X_transformed
    
    def feature_engineering(self):
        self.train_test_split()
        
        self.skin_tone_one_hot_train, self.skin_tone_one_hot_val = self.one_hot_encoding(cols = ['skin_tone'])
        self.skin_type_one_hot_train, self.skin_type_one_hot_val = self.one_hot_encoding(cols = ['skin_type'])
        
        self.tone_type_cross_train = self.cross_one_hot_features(
            one_hot_col1= self.skin_type_one_hot_train, 
            one_hot_col2= self.skin_tone_one_hot_train,
            data = self.train_X
            )
        self.tone_type_cross_val = self.cross_one_hot_features(
            one_hot_col1= self.skin_type_one_hot_val, 
            one_hot_col2= self.skin_tone_one_hot_val,
            data = self.val_X
            )

        self.hair_eye_one_hot_train, self.hair_eye_one_hot_val = self.one_hot_encoding(cols = ['hair_color', 'eye_color'])

        self.train_X_transformed = pd.concat([
            self.skin_tone_one_hot_train,
            self.skin_type_one_hot_train,
            self.tone_type_cross_train,
            self.hair_eye_one_hot_train
        ], axis= 1)

        self.val_X_transformed = pd.concat([
            self.skin_tone_one_hot_val,
            self.skin_type_one_hot_val,
            self.tone_type_cross_val,
            self.hair_eye_one_hot_val
        ], axis= 1)

        self.train_X_transformed, self.val_X_transformed = self.concat_all_features(self.train_X_transformed, self.val_X_transformed)

        return self.train_X_transformed, self.val_X_transformed



        


def plot_countbar(data: pd.DataFrame, product: str, col1: str):
    """
    given a product, return and save a countplot with column "col1" in data
    """
    plt.figure(figsize=(10,8))
    sns.countplot(x = data[col1], order = data[col1].value_counts().index)
    plt.xticks(rotation=90)
    plt.xlabel(f'Reviewers by {col1}')
    plt.savefig(f'plots/{product}/reviewers_by_{col1}.jpeg')


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
    if os.path.isdir(f'plots/{product}'):
        plt.savefig(fname = f'plots/{product}/{col1}_diff_in_means.jpeg')
    else:
        os.makedirs(f'plots/{product}')
        plt.savefig(fname = f'plots/{product}/{col1}_diff_in_means.jpeg')
    


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
    if os.path.isdir(f'plots/{product}'):
        plt.savefig(fname = f'plots/{product}/{col1}_{col2}_crosstab_heatmap.jpeg')
    else:
        os.makedirs(f'plots/{product}')
        plt.savefig(fname = f'plots/{product}/{col1}_{col2}_crosstab_heatmap.jpeg')
    
    

def plot_auc_roc(predict_y, val_y, product: str, model: str, filename: str = "auc_roc"):
    """
    given predicted probability (predict_y) and label (val_y), computes AUC-ROC across threshold and saves the figure by product and model
    """

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
    
    if os.path.isdir(f'plots/{product}'):
        plt.savefig(f'plots/{product}/{model}_{filename}.jpeg')
    else:
        os.makedirs(f'plots/{product}')
        plt.savefig(f'plots/{product}/{model}_{filename}.jpeg')


def plot_recision_recall_f1(predict_y, val_y, product: str, model: str, filename: str = "precision_recall_f1"):
    """
    given predicted probability (predict_y) and label (val_y), plots precision, recall, and F1 scores across threshold
    and saves the figure by product and model
    """

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
    
    if os.path.isdir(f'plots/{product}'):
        plt.savefig(f'plots/{product}/{model}_{filename}.jpeg')
    else:
        os.makedirs(f'plots/{product}')
        plt.savefig(f'plots/{product}/{model}_{filename}.jpeg')
    

def plot_predictions_by_scores(predict_y, val_y, product: str, model: str, bins: int = 20, filename: str = 'predictions_by_scores'):
    """
    given predicted probability (predict_y) and label (val_y), plots positive and negative instances by predicted probability
    green: label = 1; red: label = 0
    """
    plt.hist(predict_y[val_y == 1.0], bins = bins, color = 'g')
    plt.hist(predict_y[val_y == 0.0], bins = bins, color = 'r')
    plt.show()

    if os.path.isdir(f'plots/{product}'):
        plt.savefig(f'plots/{product}/{model}_{filename}.jpeg')
    else:
        os.makedirs(f'plots/{product}')
        plt.savefig(f'plots/{product}/{model}_{filename}.jpeg')
    