import pandas as pd
from time import sleep # control the crawl rate to avoid hammering the servers with too many requests
from random import randint
from tqdm import tqdm
import re

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities # dealing with pop-up permissions
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

class scrapping_foundations():
    
    def __init__(self):
        reviewer_id = []
        rating = []
        recommended = []
        review_subject = []
        review_content = []
        reviewer_feature = []
        purchased_shade = []
        date_of_review = []

        cols = {
            "reviewer_id": reviewer_id, "rating": rating, "recommended": recommended,
            "review_subject": review_subject, "review_content": review_content, 
            "reviewer_feature": reviewer_feature, "purchased_shade": purchased_shade, 
            "date_of_review": date_of_review
            }
        self.data = pd.DataFrame(cols)
        self.list_of_reviews = []
    
    def set_up_driver(self, url, popup_blocked = True, first_scroll: int = 2000, second_scroll: int = 3000):
        if popup_blocked:
            geoBlocked = webdriver.FirefoxOptions()
            geoBlocked.set_preference("geo.prompt.testing", True)
            geoBlocked.set_preference("geo.prompt.testing.allow", False)
            geoBlocked.set_preference("dom.push.enabled", False)
            self.driver = webdriver.Firefox(options=geoBlocked)
        else:
            self.driver = webdriver.Firefox()
        
        self.driver.get(url)
        sleep(3)
        self.driver.execute_script(f"window.scrollTo(0, {first_scroll})") 
        sleep(2)
        self.driver.execute_script(f"window.scrollTo(0, {second_scroll})")
        if self.driver.find_elements(By.CSS_SELECTOR, 'ul[data-comp = "Pagination StyledComponent BaseComponent "]') != []:
            self.list_of_reviews = self.driver.find_elements(By.CSS_SELECTOR, 'div[data-comp="Review StyledComponent BaseComponent "]')
        else:
            self.list_of_reviews = []
        # self.driver.find_elements_by_css_selector('div[data-comp="Review StyledComponent BaseComponent "]')
        # return list_of_reviews

    def find_end_page(self):
        page = self.driver.find_element(By.CSS_SELECTOR, 'ul[data-comp="Pagination StyledComponent BaseComponent "]')
        all_children = page.find_elements(By.CSS_SELECTOR, 'li')
        return int(all_children[-2].text)
        # if self.driver.find_elements(By.CSS_SELECTOR, 'ul[data-comp = "Pagination StyledComponent BaseComponent "] :nth-child(8)') != []:
        #     return int(self.driver.find_element(By.CSS_SELECTOR, 'ul[data-comp = "Pagination StyledComponent BaseComponent "] :nth-child(8)').text)
        
    def append_new_data(self):
        shade = ""
        review_subject = ""
        review_content = ""
        reviewer_feature = ""
        reviewer_id = ""
        rating = int()
        recommended = ""
        date_of_review = ""

        for i in self.list_of_reviews:

            if i.find_elements(By.CSS_SELECTOR, 'a[data-at="nickname"]') != []:
                reviewer_id = i.find_elements(By.CSS_SELECTOR,'a[data-at="nickname"]')[0].text
            else:
                pass 

            if i.find_elements(By.CSS_SELECTOR,'a[data-at="nickname"]+span') != []:
                reviewer_feature = i.find_elements(By.CSS_SELECTOR,'a[data-at="nickname"]+span')[0].text
            else:
                pass

            if i.find_elements(By.CSS_SELECTOR,'div[data-comp="StarRating "]') != []:
                rating = int(i.find_elements(By.CSS_SELECTOR, 'div[data-comp="StarRating "]')[0].get_attribute("aria-label")[0])
            else:
                pass

            if i.find_elements(By.CSS_SELECTOR, 'h3') != []:
                review_subject = i.find_element(By.CSS_SELECTOR, 'h3').text
            else:
                review_subject = None

            if i.find_elements(By.CSS_SELECTOR, 'img[src*="https://www.sephora.com/productimages/"]+span') != []:
                shade = i.find_elements(By.CSS_SELECTOR, 'img[src*="https://www.sephora.com/productimages/"]+span')[0].text
            else:
                shade = None
            
            if shade:
                if i.find_elements(By.CSS_SELECTOR, 'h3+div+div') != []:
                    review_content = i.find_element(By.CSS_SELECTOR, 'h3+div+div').text
                else:
                    pass
            elif i.find_elements(By.CSS_SELECTOR, 'h3+div') != []:
                review_content = i.find_element(By.CSS_SELECTOR, 'h3+div').text
            else:
                pass

            if i.find_elements(By.CSS_SELECTOR, 'span[data-at="time_posted"]') != []:
                date_of_review = i.find_element(By.CSS_SELECTOR, 'span[data-at="time_posted"]').text
            else:
                date_of_review = None

            if 'Recommended' in i.find_element(By.CSS_SELECTOR, 'div').get_attribute('innerHTML'):
                recommended = 1
            else:
                recommended = 0

            new_col = {
                "reviewer_id": reviewer_id,
                "rating": rating,
                "recommended": recommended,
                "review_subject": review_subject,
                "review_content": review_content,
                "reviewer_feature": reviewer_feature,
                "purchased_shade": shade,
                "date_of_review": date_of_review
            }
            
            self.data = self.data.append(new_col,ignore_index=True)
        
    
    def clicking_and_scraping(self, starting_page: int = 0):
        if self.list_of_reviews != []:
            end_page = self.find_end_page()
            for n in tqdm(range(starting_page, end_page)):
                self.list_of_reviews = self.driver.find_elements(By.CSS_SELECTOR, 'div[data-comp="Review StyledComponent BaseComponent "]')
                self.append_new_data()
                sleep(randint(1,3))
                self.driver.find_element(By.CSS_SELECTOR, 'button[aria-label="Next"]').click()
                # self.driver.find_element(By.CSS_SELECTOR, 'button[aria-label="Next"]').click()
                sleep(randint(2,6))
        else:
            self.data = pd.DataFrame()

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