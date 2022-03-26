import pandas as pd
from time import sleep # control the crawl rate to avoid hammering the servers with too many requests
from random import randint
from tqdm import tqdm
import re
import datetime

from google.cloud import storage

class cleaning_review_data():

    def __init__(self, filename: str) -> None:
        self.data = pd.read_csv(f'data_full_review/{filename}')

        self.data = self.data.dropna(axis = 0, subset=['reviewer_feature', 'reviewer_id', 'rating', 'date_of_review', 'review_content'], how='any')
        self.data = self.data.reset_index()
        self.filename = filename
        self.storage_client = storage.Client.from_service_account_json('foundation-matching-9bb2587b610a.json')
        
    def parsing_reviewer_features(self):
        self.data['eye_color'] = ''
        self.data['hair_color'] = ''
        self.data['skin_tone'] = ''
        self.data['skin_type'] = ''
        self.data['skin_tone_bin'] = 0 # darker skin tone as 0, lighter as 1

        for i in tqdm(range(len(self.data))):
    
            eye = re.search(r'.+eyes', self.data.loc[i, 'reviewer_feature'])
            if eye:
                self.data.loc[i, 'eye_color'] = eye.group()
                self.data.loc[i, 'eye_color'] = self.data.loc[i, 'eye_color'].replace(' eyes', '')
                self.data.loc[i, 'reviewer_feature'] = self.data.loc[i, 'reviewer_feature'].replace(eye.group(), '')
            hair = re.search(r'.+hair', self.data.loc[i, 'reviewer_feature'])
            if hair:
                self.data.loc[i, 'hair_color'] = hair.group()
                self.data.loc[i, 'hair_color'] = self.data.loc[i, 'hair_color'].replace(' hair', '').replace(', ', '')
                self.data.loc[i, 'reviewer_feature'] = self.data.loc[i, 'reviewer_feature'].replace(hair.group(), '')
            skin_tone = re.search(r'.+skin tone', self.data.loc[i, 'reviewer_feature'])
            if skin_tone:
                self.data.loc[i, 'skin_tone'] = skin_tone.group()
                self.data.loc[i, 'skin_tone'] = self.data.loc[i, 'skin_tone'].replace(' skin tone', '').replace(', ', '')
                self.data.loc[i, 'reviewer_feature'] = self.data.loc[i, 'reviewer_feature'].replace(skin_tone.group(), '')
            skin_type = re.search(r'.+skin$', self.data.loc[i, 'reviewer_feature'])
            if skin_type:
                self.data.loc[i, 'skin_type'] = skin_type.group().replace(', ', '')
                self.data.loc[i, 'skin_type'] = self.data.loc[i, 'skin_type'].replace(' skin', '')
            
            if self.data.loc[i, 'skin_tone'] == 'Dark' or self.data.loc[i, 'skin_tone'] == 'Ebony' or self.data.loc[i, 'skin_tone'] == 'Deep':
                self.data.loc[i, 'skin_tone_cat'] = 0 # darker skin tone = 0
            else:
                self.data.loc[i, 'skin_tone_cat'] = 1 # ligher skin tone = 1

        self.data = self.data.dropna(axis = 0, subset=['skin_tone', 'skin_type', 'hair_color', 'eye_color'], how='any')

        self.data = self.data[self.data['skin_tone']!='']
        self.data = self.data[self.data['skin_type']!='']
        self.data = self.data[self.data['eye_color']!='']
        self.data = self.data[self.data['hair_color']!='']
        
        self.data = self.data.reset_index(drop = True)                

        return self.data
    
    def parsing_date_of_review(self):
        for i in self.data.index:
            if re.match('.*ago*.', self.data.loc[i, 'date_of_review']):
                self.data.loc[i, 'date_of_review'] = datetime.date.today() - datetime.timedelta(int(re.findall('([\s\d]+)', self.data.loc[i, 'date_of_review'])[0]))
            
        self.data['date_of_review'] = pd.to_datetime(self.data['date_of_review'])    
        self.data['days_since_launch'] = self.data['date_of_review'] - min(self.data['date_of_review'])
        self.data['days_since_launch'] = self.data['days_since_launch'].dt.days
        self.data['days_since_launch_scaled'] = self.data['days_since_launch'] / max(self.data['days_since_launch'])

        self.data['month_of_purchase'] = pd.DatetimeIndex(self.data['date_of_review']).month

        self.data = self.data.dropna(axis=0, subset=['days_since_launch', 'days_since_launch_scaled', 'month_of_purchase'], how='any')

        return self.data
    
    def parsing_review_content(self):
        self.data['finish'] = 0
        self.data['coverage'] = 0
        self.data['shade_match'] = 0
        self.data['gifted'] = 0
        indices = self.data.index

        for i in indices:
            if re.match(r'.*coverage|cover*.', self.data.loc[i, 'review_content']):
                if self.data.loc[i, 'rating'] >= 4:
                    self.data.loc[i, 'coverage'] = 1

        for i in indices:
            if re.match(r'.*finish|matte|natural*.', self.data.loc[i, 'review_content']):
                if self.data.loc[i, 'rating'] >= 4:
                    self.data.loc[i, 'finish'] = 1

        for i in indices:
            if re.match(r'.*shade|match*.', self.data.loc[i, 'review_content']):
                if self.data.loc[i, 'rating'] >= 4:
                    self.data.loc[i, 'shade_match'] = 1
        
        for i in indices:
            if re.match(r'.*gifted|receive|incentivize|receid|compliment*.', self.data.loc[i, 'review_content']):
                if self.data.loc[i, 'rating'] >= 4:
                    self.data.loc[i, 'gifted'] = 1
        
        return self.data

    def to_pickle(self):
        self.filename = self.filename.replace('.csv', '')
        cols = ['reviewer_id', 'rating', 'recommended', 'review_subject',
                'review_content', 'reviewer_feature', 'purchased_shade',
                'date_of_review', 'eye_color', 'hair_color', 'skin_tone', 'skin_type',
                'skin_tone_bin', 'skin_tone_cat', 'days_since_launch',
                'days_since_launch_scaled', 'month_of_purchase', 'finish', 'coverage',
                'shade_match', 'gifted']
        self.data = self.data[cols]
        return self.data.to_pickle(path=f'data_full_review_cleaned/{self.filename}.pkl')

    def to_json(self):
        self.filename = self.filename.replace('.csv', '')
        cols = ['reviewer_id', 'rating', 'recommended', 'review_subject',
                'review_content', 'reviewer_feature', 'purchased_shade',
                'date_of_review', 'eye_color', 'hair_color', 'skin_tone', 'skin_type',
                'skin_tone_bin', 'skin_tone_cat', 'days_since_launch',
                'days_since_launch_scaled', 'month_of_purchase', 'finish', 'coverage',
                'shade_match', 'gifted']
        self.data = self.data[cols]
        return self.data.to_json(f'data_full_review_cleaned/{self.filename}.json', orient='records', lines=True)

    def to_gcs(self):
        bucket = self.storage_client.get_bucket('foundation_reviews')
        blob = bucket.blob(f'{self.filename}.json')
        blob.upload_from_filename(f'data_full_review_cleaned/{self.filename}.json')
