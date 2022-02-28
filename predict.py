import pandas as pd

pd.options.mode.chained_assignment = None
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

import re

import warnings
warnings.filterwarnings('ignore')

import pickle

import xgboost as xgb
import os

class feature_engineering(object):

    def __init__(self, product_name: str, input_dict: dict):
        self.product_name = product_name
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

        self.data = pd.DataFrame([input_dict])
        self.training_data = pd.read_json(f"data_full_review_cleaned/{product_name}.json", lines=True)
        self.X = self.training_data[self.all_features]
        self.y = self.training_data['recommended']
        self.train_X, self.val_X, self.train_y, self.val_y = train_test_split(self.X, self.y, test_size=0.3,
                                                                              random_state=0)

    def dropping_outlier_reviewers(self, cols=None):
        """
        Some reviewers can be considered outliers based on their features,
        e.g., if only less than 10% of the reviewers in the dataset had gray hair, we drop these reviewers
        I define outliers by (a) being less than 5% out of the whole dataset, and
        (b) having less than 5 people in the validation set
        This function drops these reviewers in both the training and validation sets after doing train_test_split
        """
        # self.logger.info(f'Before dropping outliers:')
        # self.logger.info(f'N in train_X is {len(self.train_X)}, N in val_X is {len(self.val_X)}')

        if cols is None:
            cols = ['hair_color', 'eye_color', 'skin_tone']

        for c in cols:
            c_proportion = self.val_X.groupby([c]).count()['coverage'] / len(self.val_X)
            outliers_proportion = c_proportion[c_proportion <= 0.05].index
            # if only less than 10% of some reviewers in the validation set had a certain feature
            # like if only 5 people in the validation set out of 100 people in the complete dataset had red hair
            # we drop these people
            c_count = self.val_X.groupby([c]).count()['coverage']
            outliers_count = c_count[c_count < 5].index
            # if only less than 5 reviewers in the validation set had a certain feature
            # we drop these reviewers

            if not outliers_proportion.empty or not outliers_count.empty:
                # self.logger.info(f'#### Removing outliers in {c} ####')
                n_dropped = 0
                if not outliers_count.empty:
                    for i in outliers_count:
                        val_indices = self.val_X[self.val_X[c] == i].index
                        n_dropped += len(val_indices)
                        self.val_X.drop(index=val_indices, inplace=True)
                        self.val_y.drop(index=val_indices, inplace=True)

                        train_indices = self.train_X[self.train_X[c] == i].index
                        n_dropped += len(train_indices)
                        self.train_X.drop(index=train_indices, inplace=True)
                        self.train_y.drop(index=train_indices, inplace=True)
                        # self.train_X = self.train_X[self.train_X[c] != i]
                    # self.logger.info(
                    #     f'{n_dropped} reviewers with value {outliers_count.to_list()} in {c} are dropped'
                    # )
                    continue

                elif not outliers_proportion.empty:
                    for i in outliers_proportion:
                        val_indices = self.val_X[self.val_X[c] == i].index
                        n_dropped += len(val_indices)
                        self.val_X.drop(index=val_indices, inplace=True)
                        self.val_y.drop(index=val_indices, inplace=True)

                        train_indices = self.train_X[self.train_X[c] == i].index
                        n_dropped += len(train_indices)
                        self.train_X.drop(index=train_indices, inplace=True)
                        self.train_y.drop(index=train_indices, inplace=True)

    def one_hot_encoding(self, col: str):
        """
        one-hot encode a feature given col,
        rename the columns after one-hot encoding given the categories in col,
        and return test set as pd.DataFrame
        """
        with open(f'models/{self.product_name}/encoder_{self.product_name}_{col}.pickle', 'rb') as f:
            enc_rest = pickle.load(f)

        with open(f'models/{self.product_name}/col_names_{self.product_name}_{col}.pickle', 'rb') as f:
            col_names_dict = pickle.load(f)

        if 'color' in col:
            # the column names for eye colors and hair colors are formatted as '{color}_eye' or '{color}_hair'
            # need to match the format
            part = col.split('_')[0]
            color = self.data.loc[0, col] + '_' + part

            if color in col_names_dict.values():
                test_X_transform = enc_rest.transform(self.data[[col]])
                test_X_transform = pd.DataFrame(test_X_transform)

                test_X_transform.rename(columns=col_names_dict, inplace=True)

                return test_X_transform

            else:
                # print(f"no such category for {col}")
                return pd.DataFrame()
        else:
            # for columns other than hair or eye colors
            if self.data.loc[0, col] in col_names_dict.values():
                test_X_transform = enc_rest.transform(self.data[[col]])
                test_X_transform = pd.DataFrame(test_X_transform)

                test_X_transform.rename(columns=col_names_dict, inplace=True)

                return test_X_transform

            else:
                return pd.DataFrame()

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
        data_cross = pd.DataFrame()

        if any(one_hot_col1.notnull().sum()==1) and any(one_hot_col2.notnull().sum()==1):
            total_col1_cat = one_hot_col1.columns.to_list()
            total_col2_cat = one_hot_col2.columns.to_list()
        # total_col1_cat = data.groupby([col1], as_index=False).count()[col1]
        # total_col2_cat = data.groupby([col2], as_index=False).count()[col2]

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

        else:
            return data_cross

    def concat_all_features(
            self,
            test_X_transformed: pd.DataFrame
    ):
        """
        concatenate non-one-hot-encoded features with one-hot-encoded ones
        return test set
        """
        for i in self.other_features:
            test_X_transformed[i] = self.data[i].reset_index(drop=True)
        return test_X_transformed

    def feature_engineering(self):
        """
        put together all the steps to engineer features
        """

        self.skin_tone_one_hot_test = self.one_hot_encoding(col='skin_tone')
        self.skin_type_one_hot_test = self.one_hot_encoding(col='skin_type')

        self.tone_type_cross_test = self.cross_one_hot_features(
            one_hot_col1=self.skin_tone_one_hot_test,
            one_hot_col2=self.skin_type_one_hot_test,
            data=self.data
        )

        self.hair_one_hot_test = self.one_hot_encoding(col='hair_color')
        self.eye_one_hot_test = self.one_hot_encoding(col='eye_color')

        one_hot_cols = [
            self.skin_tone_one_hot_test,
            self.skin_type_one_hot_test,
            self.tone_type_cross_test,
            self.hair_one_hot_test,
            self.eye_one_hot_test
        ]

        if any([col.empty for col in one_hot_cols]):
            return pd.DataFrame()
        else:
            self.test_X_transformed = pd.concat([
                self.skin_tone_one_hot_test,
                self.skin_type_one_hot_test,
                self.tone_type_cross_test,
                self.hair_one_hot_test,
                self.eye_one_hot_test
            ], axis=1)

            self.test_X_transformed = self.concat_all_features(self.test_X_transformed)

            return self.test_X_transformed

def predict_from_user_input(input: dict):
    list_of_foundation_names = []
    for file in os.listdir('models/'):
        list_of_foundation_names.append(os.path.join(file))
    list_of_foundation_names = sorted(list_of_foundation_names)[1:]
    # list_of_foundation_names = list_of_foundation_names[:5]

    scores_cols = {'brand_product': str(), 'scores': float()}
    scores = pd.DataFrame([scores_cols])

    for i in range(len(list_of_foundation_names)):

        new_instance = feature_engineering(
            product_name=list_of_foundation_names[i],
            input_dict=input
        )
        new_data = new_instance.feature_engineering()

        if not new_data.empty:
            model = xgb.XGBClassifier(objective='binary:logistic', use_label_encoder=False)
            model.load_model(f'models/{list_of_foundation_names[i]}/{list_of_foundation_names[i]}_xgb.model')
            results = model.predict_proba(new_data)[:, 1]
            scores.loc[i, 'brand_product'] = list_of_foundation_names[i]
            scores.loc[i, 'scores'] = results

    scores = scores.sort_values('scores', ascending=False).reset_index(drop=True)
    scores = scores.iloc[0:4]
    return scores

def get_longest_dict(col: str):
    """
    given a feature (col) from all the review data, return the longest dictionary for all possible value for the feature
    """
    list_of_foundation_names = []
    for file in os.listdir('models/'):
        list_of_foundation_names.append(os.path.join(file))
    list_of_foundation_names = sorted(list_of_foundation_names)[1:]

    col_names_dict = dict()

    for i in range(len(list_of_foundation_names)):
        with open(
            f'models/{list_of_foundation_names[i]}/col_names_{list_of_foundation_names[i]}_{col}.pickle',
            'rb'
        ) as f:
            col_names_dict_for_i = pickle.load(f)

        for j in col_names_dict_for_i.values():
            if j not in col_names_dict.values():
                col_names_dict[len(col_names_dict)] = j
    if 'color' in col:
        part = col.split('_')[0]
        part = '_' + part
        for key, value in col_names_dict.items():
            col_names_dict[key] = value.replace(part, '')

    return col_names_dict


def filter_shade(input: dict, brand_product: str):
    training_data = pd.read_json(f'data_full_review_cleaned/{brand_product}.json', lines=True)
    training_data = training_data[training_data['rating'] == 5]

    # first filter through skin tone
    tone_data = training_data[training_data['skin_tone'] == input['skin_tone']]
    hair_data = tone_data[tone_data['hair_color'] == input['hair_color']]
    eye_data = hair_data[hair_data['eye_color'] == input['eye_color']]

    if len(eye_data['purchased_shade'].unique()) > 3:
        eye_data = eye_data.groupby('purchased_shade', as_index=False).count().sort_values('brand_product',
                                                                                           ascending=False)[:3]
        list_of_shades = eye_data['purchased_shade']
    elif len(hair_data['purchased_shade'].unique()) > 3:
        hair_data = hair_data.groupby('purchased_shade', as_index=False).count().sort_values('brand_product',
                                                                                             ascending=False)[:3]
        list_of_shades = hair_data['purchased_shade']
    else:
        tone_data = tone_data.groupby('purchased_shade', as_index=False).count().sort_values('brand_product',
                                                                                             ascending=False)[:3]
        list_of_shades = tone_data['purchased_shade']

    return list_of_shades.to_list()

def extracting_img_src(brand_product: str):
    foundation_features_parsed_url = pd.read_csv('foundation_features_parsed_url.csv')
    img_src = foundation_features_parsed_url.loc[
        foundation_features_parsed_url['brand_product'] == f'{brand_product}', 'img_src'].values[0]
    return img_src

def extracting_url(brand_product: str):
    foundation_features_parsed_url = pd.read_csv('foundation_features_parsed_url.csv')
    url = foundation_features_parsed_url.loc[
        foundation_features_parsed_url['brand_product'] == f'{brand_product}', 'url'].values[0]
    return url

