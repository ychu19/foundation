import pandas as pd
from time import sleep # control the crawl rate to avoid hammering the servers with too many requests
from random import randint
from tqdm import tqdm
import re

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities # dealing with pop-up permissions
from selenium.webdriver.firefox.options import Options

class scrapping_foundations():
    
    def __init__(self):
        reviewer_id = []
        rating = []
        review_subject = []
        reviewer_feature = []
        purchased_shade = []

        cols = {
            "reviewer_id": reviewer_id, "rating": rating, "review_subject": review_subject,
            "reviewer_feature": reviewer_feature, "purchased_shade": purchased_shade
            }
        self.data = pd.DataFrame(cols)
        self.list_of_revies = []
    
    def set_up_driver(self, url, geoBlocked = True, first_scroll: int = 2000, second_scroll: int = 3000):
        if geoBlocked:
            geoBlocked = webdriver.FirefoxOptions()
            geoBlocked.set_preference("geo.prompt.testing", True)
            geoBlocked.set_preference("geo.prompt.testing.allow", False)
            self.driver = webdriver.Firefox(options=geoBlocked)
        else:
            self.driver = webdriver.Firefox()
        
        self.driver.get(url)
        sleep(3)
        self.driver.execute_script(f"window.scrollTo(0, {first_scroll})") 
        sleep(2)
        self.driver.execute_script(f"window.scrollTo(0, {second_scroll})") 
        self.list_of_reviews = self.driver.find_elements_by_css_selector('div[data-comp="Review StyledComponent BaseComponent "]')
        # return list_of_reviews

    def append_new_data(self):
        shade = ""
        review_subject = ""
        reviewer_feature = ""
        reviewer_id = ""
        rating = int()
        for i in self.list_of_reviews:
            
            if i.find_elements_by_css_selector('strong[data-at="nickname"]') != []:
                reviewer_id = i.find_elements_by_css_selector('strong[data-at="nickname"]')[0].text
            else:
                pass 
            
            if i.find_elements_by_css_selector('strong[data-at="nickname"]+span') != []:
                reviewer_feature = i.find_elements_by_css_selector('strong[data-at="nickname"]+span')[0].text
            else:
                pass

            if i.find_elements_by_css_selector('div[data-comp="StarRating "]') != []:
                rating = int(i.find_elements_by_css_selector('div[data-comp="StarRating "]')[0].get_attribute("aria-label")[0])
            else:
                pass

            if i.find_elements_by_css_selector('h3') != []:
                review_subject = i.find_element_by_css_selector('h3').text
            else:
                review_subject = None

            if i.find_elements_by_css_selector('img[src*="https://www.sephora.com/productimages/"]+span') != []:
                shade = i.find_elements_by_css_selector('img[src*="https://www.sephora.com/productimages/"]+span')[0].text
            else:
                shade = None

            new_col = {
                "reviewer_id": reviewer_id,
                "rating": rating,
                "review_subject": review_subject,
                "reviewer_feature": reviewer_feature,
                "purchased_shade": shade
            }
            # print(new_col)
            self.data = self.data.append(new_col,ignore_index=True)
        # return self.data
    
    def clicking_and_scraping(self, starting_page: int = 0, ending_page: int = 201):
        for n in tqdm(range(starting_page, ending_page)):
            self.list_of_reviews = self.driver.find_elements_by_css_selector('div[data-comp="Review StyledComponent BaseComponent "]')
            self.append_new_data()
            self.driver.find_element_by_css_selector('button[aria-label="Next"]').click()
            sleep(randint(2,6))

    def close_driver(self):
        self.driver.close()


class cleaning_review_data():

    def __init__(self, path: str) -> None:
        self.data = pd.read_csv(path)

        self.data = self.data.dropna(axis = 0, subset=['reviewer_feature', 'reviewer_id', 'rating'], how='any')
        self.data = self.data.reset_index()
        
    def parsing_reviewer_features(self):
        self.data['eye_color'] = ''
        self.data['hair_color'] = ''
        self.data['skin_tone'] = ''
        self.data['skin_type'] = ''

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
        
        return self.data
    
    def to_pickle(self, file_name: str):
        return self.data.to_pickle(path=f'data_cleaned/{file_name}.pkl')