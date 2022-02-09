import pandas as pd

pd.options.mode.chained_assignment = None
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import (
    f1_score,
    auc,
    precision_score,
    recall_score,
    accuracy_score
)

import os
import re

import scipy.stats as st

import xgboost as xgb
import lightgbm as lgb
import optuna
from optuna.samplers import RandomSampler, TPESampler
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error, log_loss, roc_auc_score, accuracy_score, precision_score, recall_score, \
    f1_score


def create_dir_for_product_if_not_existent(product, plots_or_models: str = 'plots'):
    """
    check if directory 'plots/{product}' exists
    if true, save the fig in the directory as {filename_}.jpeg
    if false, create the directory and save the fig in the newly created directory
    """
    if plots_or_models == 'plots':
        if os.path.isdir(f'{plots_or_models}/{product}'):
            pass
            # plt.savefig(fname=f'{plots_or_models}/{product}/{filename_}.jpeg')
        else:
            os.makedirs(f'{plots_or_models}/{product}')
            # plt.savefig(fname=f'{plots_or_models}/{product}/{filename_}.jpeg')
    if plots_or_models == 'models':
        if os.path.isdir(f'{plots_or_models}/{product}'):
            pass
            # model.save_model(fname=f'{plots_or_models}/{product}/{filename_}.model')
        else:
            os.makedirs(f'{plots_or_models}/{product}')
            # model.save_model(fname=f'{plots_or_models}/{product}/{filename_}.model')


class feature_engineering(object):
    """
    loads data from pickle by product_name.json, split the data into train and validation sets,
    one-hot encode reviewer features (eye color, hair color, skin color, and skin tone),
    cross one-hot encoded skin color and skin tone,
    combine the crossed features, one-hot encoded features, and other features into two pd.DataFrames,
    one is the training set with all features, and the other is the validation set with all features
    return the training and validation sets
    """

    def __init__(self, file_name: str) -> None:
        self.data = pd.read_json(f'data_full_review_cleaned/{file_name}', lines=True)
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
        self.product_name = file_name.replace('.json', '')
        # perhaps some logging to know if there's class imbalance issues?

    def train_test_split(self, proportion: float = 0.3):
        """
        split training and validation sets given proportion using train_test_split() from scikit-learn
        """
        self.train_X, self.val_X, self.train_y, self.val_y = train_test_split(self.X, self.y, test_size=proportion,
                                                                              random_state=0)

    def one_hot_encoding(self, col: str):
        """
        one-hot encode a feature given col,
        rename the columns after one-hot encoding given the categories in col,
        and return training and validation sets as pd.DataFrame
        """
        enc_rest = OneHotEncoder(sparse=False)

        train_X_transform = enc_rest.fit_transform(self.train_X[[col]])
        val_X_transform = enc_rest.transform(self.val_X[[col]])

        train_X_transform = pd.DataFrame(train_X_transform)
        # print(train_X_transform.columns)
        col_names_dict = dict()

        cols_cat = self.data.groupby([col], as_index=False).count()[col]
        if re.match('^.*_color$', col):
            color = re.split('_color', col)[0]
            cols_cat = cols_cat.str.cat(pd.Series([color] * len(self.data)), sep='_')

        for col_idx in train_X_transform.columns:
            col_names_dict[col_idx] = cols_cat[col_idx]
        train_X_transform.rename(columns=col_names_dict, inplace=True)

        val_X_transform = pd.DataFrame(val_X_transform)
        col_names_dict = dict()
        for col_idx in val_X_transform.columns:
            col_names_dict[col_idx] = cols_cat[col_idx]
        val_X_transform.rename(columns=col_names_dict, inplace=True)

        return train_X_transform, val_X_transform

    def cross_one_hot_features(
            self,
            one_hot_col1: pd.DataFrame,
            one_hot_col2: pd.DataFrame,
            data: pd.DataFrame,
            col1: str = 'skin_type',
            col2: str = 'skin_tone'
    ):
        """
        given one-hot encoded feature 1 (one_hot_col1) and one-hot encoded feature 2 (one_hot_col1),
        returns a dataframe with crossed feature b/w one_hot_col1 and one_hot_col2
        columns of the returned dataframe are named by each of the crossed categories in col1 and col2 as "col1_col2"
        """
        total_col1_cat = data.groupby([col1], as_index=False).count()[col1]
        total_col2_cat = data.groupby([col2], as_index=False).count()[col2]

        data_cross = pd.DataFrame()

        i = 0  # col1

        while i <= len(total_col1_cat) - 1:
            j = 0  # col2
            while j <= len(total_col2_cat) - 1:
                col1_cat = total_col1_cat[i]
                col2_cat = total_col2_cat[j]
                new_cross = one_hot_col1[col1_cat] * one_hot_col2[col2_cat]
                new_cross = pd.Series(new_cross)
                data_cross[f'{col1_cat}_{col2_cat}'] = new_cross
                j += 1
            i += 1

        return data_cross.dropna(axis=0)

    def concat_all_features(
            self,
            train_X_transformed: pd.DataFrame,
            val_X_transformed: pd.DataFrame
    ):
        """
        concatenate non-one-hot-encoded features with one-hot-encoded ones
        return training and validation sets
        """
        for i in self.other_features:
            train_X_transformed[i] = self.train_X[i].reset_index(drop=True)
            val_X_transformed[i] = self.val_X[i].reset_index(drop=True)
        return train_X_transformed, val_X_transformed

    def feature_engineering(self):
        self.train_test_split()

        self.skin_tone_one_hot_train, self.skin_tone_one_hot_val = self.one_hot_encoding(col='skin_tone')
        self.skin_type_one_hot_train, self.skin_type_one_hot_val = self.one_hot_encoding(col='skin_type')

        self.tone_type_cross_train = self.cross_one_hot_features(
            one_hot_col1=self.skin_type_one_hot_train,
            one_hot_col2=self.skin_tone_one_hot_train,
            data=self.train_X
        )
        self.tone_type_cross_val = self.cross_one_hot_features(
            one_hot_col1=self.skin_type_one_hot_val,
            one_hot_col2=self.skin_tone_one_hot_val,
            data=self.train_X
            # need to use train_X for categories, in case that val_X.groupby([col]) skips categories with zero counts
        )

        self.hair_one_hot_train, self.hair_one_hot_val = self.one_hot_encoding(col='hair_color')
        self.eye_one_hot_train, self.eye_one_hot_val = self.one_hot_encoding(col='eye_color')

        self.train_X_transformed = pd.concat([
            self.skin_tone_one_hot_train,
            self.skin_type_one_hot_train,
            self.tone_type_cross_train,
            self.hair_one_hot_train,
            self.eye_one_hot_train
        ], axis=1)

        self.val_X_transformed = pd.concat([
            self.skin_tone_one_hot_val,
            self.skin_type_one_hot_val,
            self.tone_type_cross_val,
            self.hair_one_hot_val,
            self.eye_one_hot_val
        ], axis=1)

        self.train_X_transformed, self.val_X_transformed = self.concat_all_features(self.train_X_transformed,
                                                                                    self.val_X_transformed)

        return None


class training_models(feature_engineering):

    def __init__(self, file_name, train_X_transformed, val_X_transformed, train_y, val_y):
        feature_engineering.__init__(self, file_name)
        self.train_X_transformed = train_X_transformed
        self.val_X_transformed = val_X_transformed
        self.train_y = train_y
        self.val_y = val_y
        self.product_name = file_name.replace('.json', '')

    def create_DMatrix_for_xgb(self):
        """
        given the training and validation sets,
        calculate 'balanced' class weights,
        transform the training and validation sets into xgb.DMatrix
        """
        train_p_1 = len(self.train_y) / (len(self.train_y[self.train_y == 1]) * 2)
        train_p_0 = len(self.train_y) / (len(self.train_y[self.train_y == 0]) * 2)

        val_p_1 = len(self.val_y) / (len(self.val_y[self.val_y == 1]) * 2)
        val_p_0 = len(self.val_y) / (len(self.val_y[self.val_y == 0]) * 2)

        self.w_train = np.where(self.train_y == 0, train_p_0, train_p_1)
        self.w_val = np.where(self.val_y == 0, val_p_0, val_p_1)

        self.dtrain = xgb.DMatrix(self.train_X_transformed, label=self.train_y, weight=self.w_train)
        self.dval = xgb.DMatrix(self.val_X_transformed, label=self.val_y, weight=self.w_val)

        return self.dtrain, self.dval

    def objective(
            self,
            trial: optuna.Trial
    ) -> float:
        """
        The objective function calculated while tuning hyperparameters.
        The suggestions for the next set of hyperparameters on the next round is
        performed using Tree Structured Parzen Estimation. The value being optimized
        for is mse.
        Returns the mse value for the trial calculated on the test set
        """
        hyperparams = {
            'verbosity': 0,
            'booster': trial.suggest_categorical("booster", ['gbtree', 'gblinear', 'dart']),
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'lambda': trial.suggest_loguniform('lambda', 1e-8, 1.0),
            'alpha': trial.suggest_loguniform('alpha', 1e-8, 1.0),
        }

        if hyperparams["booster"] == "gbtree" or hyperparams["booster"] == "dart":
            hyperparams["max_depth"] = trial.suggest_int("max_depth", 1, 15)
            hyperparams["eta"] = trial.suggest_loguniform("eta", 1e-8, 1.0)
            hyperparams["gamma"] = trial.suggest_loguniform("gamma", 1e-8, 1.0)
            hyperparams["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])

        if hyperparams["booster"] == "dart":
            hyperparams["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
            hyperparams["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
            hyperparams["rate_drop"] = trial.suggest_loguniform("rate_drop", 1e-8, 1.0)
            hyperparams["skip_drop"] = trial.suggest_loguniform("skip_drop", 1e-8, 1.0)

        pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "validation-auc")
        bst = xgb.train(hyperparams, self.dtrain, evals=[(self.dval, "validation"), (self.dtrain, 'train')],
                        callbacks=[pruning_callback], verbose_eval=False)
        predicts = bst.predict(self.dval)
        predict_y = np.rint(predicts)
        accuracy = accuracy_score(self.val_y, predict_y)

        return accuracy

    def hyperparameter_tune(
            self,
            # N_TRIALS: int
    ) -> dict:
        """
        This function performs the hyperparameter tuning given a training dataset and
        a test dataset.
        Returns the best hyperparameter set
        """
        initial_hyperparams = {
            'verbosity': 0,
            # 'booster': trial.suggest_categorical("booster", ['gbtree', 'gblinear', 'dart']),
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            # 'lambda': trial.suggest_loguniform('lambda', 1e-8, 1.0),
            # 'alpha': trial.suggest_loguniform('alpha', 1e-8, 1.0),
        }

        study = optuna.create_study(
            study_name=self.product_name,
            direction='minimize',
            sampler=TPESampler()
            # sampler=RandomSampler()
        )

        study.optimize(
            lambda trial: self.objective(trial),
            # timeout=HYPERPARAM_TIMEOUT,
            n_trials=10,
            n_jobs=1
        )
        tuned_params = study.best_trial.params

        # Replace guessed parameter with concrete best values
        for key, value in tuned_params.items():
            initial_hyperparams[key] = value

        return initial_hyperparams

    def train_single_run(
            self,
            parameters: dict,
            num_round: int
    ) -> xgb.train:
        """
        Accepts a set of hyperparameters, a training dataset and a parameters dict
        and returns a trained model on this data. Note that both early stopping and
        the metric calculation is determined based on trying to maximize the f1score
        on the test data, which has the native imbalanced distribution

        Return a lightgbm.Booster object
        """

        evallist = [(self.dval, 'eval'), (self.dtrain, 'train')]
        model = xgb.train(parameters, self.dtrain, num_round, evallist, verbose_eval=False)

        return model

    def training_xgb_model(
            self,
            # N_TRIALS:int=10
    ):
        self.dtrain, self.dval = self.create_DMatrix_for_xgb()

        params = self.hyperparameter_tune()

        best_model = self.train_single_run(
            parameters=params,
            num_round=10
        )

        predicts = best_model.predict(self.dval)
        # print(predicts)
        self.predict_y = [round(value) for value in predicts]

        create_dir_for_product_if_not_existent(product=self.product_name, plots_or_models='models')
        best_model.save_model(f'models/{self.product_name}/{self.product_name}_xgb.model')


def plot_countbar(data: pd.DataFrame, product: str, col1: str):
    """
    given a product, return and save a countplot with column "col1" in data
    """
    plt.figure(figsize=(10, 8))
    sns.countplot(x=data[col1], order=data[col1].value_counts().index)
    plt.xticks(rotation=90)
    plt.xlabel(f'Reviewers by {col1}')
    plt.title(product.replace('_', ' '))

    create_dir_for_product_if_not_existent(product=product, plots_or_models='plots')
    plt.savefig(f'plots/{product}/{col1}_countbar.jpeg')


def plot_line(data: pd.DataFrame, product: str, col1: str):
    """
    given a product, return and save a line plot with continuous var "col1" in data
    """
    plt.plot(data[col1])
    plt.title(product.replace('_', ' '))

    create_dir_for_product_if_not_existent(product=product, plots_or_models='plots')
    plt.savefig(f'plots/{product}/{col1}_lineplot.jpeg')


def plot_diff_in_means(data: pd.DataFrame, product: str, col1: str, col2: str = 'rating'):
    """
    given a product, plots difference in means across groups and saves the figure
    col1
        categorical data with groups
    col2
        continuous data for the means
    """

    data_groupby = data.groupby(col1, as_index=False)[col2]
    data_groupby_labels = data_groupby.agg('mean').sort_values(col2, ascending=False)

    data_agg = pd.DataFrame(data_groupby_labels).rename(columns={col2: 'mean'})

    n = data_groupby.count()
    data_agg = data_agg.merge(n, how='left', right_on=col1, left_on=col1).rename(columns={col2: 'n'})

    std = data_groupby.agg(np.std)
    data_agg = data_agg.merge(std, how='left', right_on=col1, left_on=col1).rename(columns={col2: 'std'})
    data_agg['se'] = data_agg['std'] / np.sqrt(data_agg['n'])

    data_agg['lower'] = st.t.interval(alpha=0.95, df=data_agg['n'] - 1, loc=data_agg['mean'], scale=data_agg['se'])[0]
    data_agg['upper'] = st.t.interval(alpha=0.95, df=data_agg['n'] - 1, loc=data_agg['mean'], scale=data_agg['se'])[1]

    for upper, mean, lower, y in zip(data_agg['upper'], data_agg['mean'], data_agg['lower'], data_agg[col1]):
        plt.plot((lower, mean, upper), (y, y, y), 'b.-')
    plt.yticks(range(len(n)), list(data_agg[col1]))

    create_dir_for_product_if_not_existent(product=product, plots_or_models='plots')
    plt.savefig(f'plots/{product}/{col1}_{col2}_diff_in_means.jpeg')


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
    col1_col2_crosstab = pd.crosstab(data[col1], data[col2], values=data[col3], aggfunc=func)
    col1_labels = data.groupby(col1, as_index=False).count()[col1]
    col2_labels = data.groupby(col2, as_index=False).count()[col2]

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(col1_col2_crosstab)
    ax.set_yticks(np.arange(len(col1_labels)), labels=col1_labels)
    ax.set_xticks(np.arange(len(col2_labels)), labels=col2_labels)
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(col3, rotation=-90, va="bottom")

    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    # threshold = im.norm(col1_col2_crosstab.max())/2
    texts = []
    textcolors = ("black", "white")
    for i in range(col1_col2_crosstab.shape[0]):
        for j in range(col1_col2_crosstab.shape[1]):
            kw.update(color=textcolors[int(im.norm(col1_col2_crosstab.iloc[i, j]) < 0.5)])
            text = im.axes.text(j, i, round(col1_col2_crosstab.iloc[i, j], 2), kw)
            texts.append(text)
    plt.title(product.replace('_', ' '))

    create_dir_for_product_if_not_existent(product=product, plots_or_models='plots')
    plt.savefig(f'plots/{product}/{col1}_{col2}_crosstab_heatmap.jpeg')


def plot_auc_roc(predict_y, val_y, product: str, model: str, filename: str = "auc_roc"):
    """
    given predicted probability (predict_y) and label (val_y), computes AUC-ROC across threshold and saves the figure by product and model
    """

    thresholds = np.linspace(0, 1, 10)

    fpr = []
    tpr = []

    for threshold in thresholds:
        predict_y_binary = np.where(predict_y >= threshold, 1, 0)

        fp = np.sum((val_y == 0) & (predict_y_binary == 1))  # true value is 0 but predict to be 1
        tp = np.sum((val_y == 1) & (predict_y_binary == 1))  # true value is 1 & predict to be 1

        fn = np.sum((val_y == 1) & (predict_y_binary == 0))  # true value is 1 but predict to be 0
        tn = np.sum((val_y == 0) & (predict_y_binary == 0))  # true value is 0 but predict to be 0

        fpr.append(fp / (fp + tn))
        tpr.append(tp / (tp + fn))

        ## ROC & ROC-AUC
    roc_auc_ = round(auc(fpr, tpr), 5)
    fig, ax3 = plt.subplots(1, 1)
    ax3.plot(fpr, tpr, label="ROC")
    ax3.plot([0, 1], [0, 1], 'k--')
    ax3.set_xlabel("False Positive Rate")
    ax3.set_ylabel("True Positive Rate")
    ax3.text(0.55, 0.2, 'AUC = {}'.format(roc_auc_))
    product_title = product.replace('_', ' ')
    plt.title(f'{model} with {product_title}')

    create_dir_for_product_if_not_existent(product=product, plots_or_models='plots')
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

    precision_at_threshold = precision_score(val_y, np.where(predict_y >= 0.5, 1, 0))
    recall_at_threshold = recall_score(val_y, np.where(predict_y >= 0.5, 1, 0))
    f1_score_at_threshold = f1_score(val_y, np.where(predict_y >= 0.5, 1, 0))

    ### plot precisions, recalls, and F1 scores across threshold
    fig, ax2 = plt.subplots(1, 1)
    ax2.plot(thresholds, precisions, label="precision")
    ax2.plot(thresholds, recalls, label="recall")
    ax2.plot(thresholds, f1_scores, label="f1 score")
    ax2.axvline(0.5, linestyle=':', color='r')
    ax2.plot(0.5, precision_at_threshold, "s")
    ax2.annotate("{}".format(round(precision_at_threshold, 3)), (0.5, precision_at_threshold))
    ax2.plot(0.5, recall_at_threshold, "s")
    ax2.annotate("{}".format(round(recall_at_threshold, 3)), (0.5, recall_at_threshold))
    ax2.plot(0.5, f1_score_at_threshold, "s")
    ax2.annotate("{}".format(round(f1_score_at_threshold, 3)), (0.5, f1_score_at_threshold))
    ax2.legend()
    ax2.set_xlabel("Thresholds")

    product_title = product.replace('_', ' ')
    plt.title(f'{model} with {product_title}')

    create_dir_for_product_if_not_existent(product=product, plots_or_models='plots')
    plt.savefig(f'plots/{product}/{model}_{filename}.jpeg')


def plot_predictions_by_scores(predict_y, val_y, product: str, model: str, bins: int = 20,
                               filename: str = 'predictions_by_scores'):
    """
    given predicted probability (predict_y) and label (val_y), plots positive and negative instances by predicted probability
    green: label = 1; red: label = 0
    """
    plt.hist(predict_y[val_y == 1.0], bins=bins, color='g')
    plt.hist(predict_y[val_y == 0.0], bins=bins, color='r')

    product_title = product.replace('_', ' ')
    plt.title(f'{model} with {product_title}')

    plt.show()

    create_dir_for_product_if_not_existent(product=product, plots_or_models='plots')
    plt.savefig(f'plots/{product}/{model}_{filename}.jpeg')
