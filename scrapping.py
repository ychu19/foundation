import pandas as pd
from time import sleep # control the crawl rate to avoid hammering the servers with too many requests
from random import randint
from tqdm import tqdm
import re
import datetime

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities # dealing with pop-up permissions
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from google.cloud import storage

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
        sleep(1)
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
                button = self.driver.find_element(By.CSS_SELECTOR, 'button[aria-label="Next"]')
                button.click()
                # self.driver.find_element(By.CSS_SELECTOR, 'button[aria-label="Next"]').click()
                sleep(randint(2,6))
        else:
            self.data = pd.DataFrame()

    def close_driver(self):
        self.driver.close()

class scrapping_foundation_features(object):

    def __init__(self, url: str, product_name: str):
        self.url = url
        self.product_name = product_name
        self.product_features = []
        self.product_description = []
        self.storage_client = storage.Client.from_service_account_json('foundation-matching-9bb2587b610a.json')

    def set_up_driver(self, popup_blocked=True, scroll: int = 600):
        if popup_blocked:
            geoBlocked = webdriver.FirefoxOptions()
            geoBlocked.set_preference("geo.prompt.testing", True)
            geoBlocked.set_preference("geo.prompt.testing.allow", False)
            geoBlocked.set_preference("dom.push.enabled", False)
            self.driver = webdriver.Firefox(options=geoBlocked)
        else:
            self.driver = webdriver.Firefox()

        self.driver.get(self.url)
        sleep(3)
        self.driver.execute_script(f"window.scrollTo(0, {scroll})")

    def scrap_product_description_and_features(self):
        if self.driver.find_elements(By.XPATH, '//h2[contains(text(), "Highlights")]/following-sibling::div') != []:
            self.product_features = self.driver.find_elements(By.XPATH, '//h2[contains(text(), "Highlights")]/following-sibling::div')[0].text
        else:
            pass
        if self.driver.find_elements(By.XPATH, '//h2[contains(text(), "About the Product")]/following-sibling::div') != []:
            self.product_description = self.driver.find_elements(By.XPATH, '//h2[contains(text(), "About the Product")]/following-sibling::div')[1].text
        else:
            pass

    def scrap_product_price(self):
        if self.driver.find_elements(By.CSS_SELECTOR, 'p[data-comp="Price "'):
            self.price = self.driver.find_elements(By.CSS_SELECTOR, 'p[data-comp="Price "')[0].text.split('\n')[0]
        else:
            self.price = 0
        return self.price


    def save_data_to_json(self):
        cols = {
            "brand_product": self.product_name,
            "product_features": self.product_features,
            "product_description": self.product_description,
            "price": self.price
        }
        self.data = pd.DataFrame([cols])
        self.data.to_json(f'data_foundation_features/{self.product_name}_features.json', orient='records', lines=True)

    def push_data_to_gsc(self):
        bucket = self.storage_client.get_bucket('foundation_features')
        blob = bucket.blob(f'{self.product_name}.json')
        blob.upload_from_filename(f'data_foundation_features/{self.product_name}_features.json')

    def close_driver(self):
        self.driver.close()

class scrapping_foundation_images(object):

    def __init__(self, url: str, product_name: str):
        self.url = url
        self.product_name = product_name
        self.img_url = str()

    def set_up_driver(self, popup_blocked=True, scroll: int = 600):
        if popup_blocked:
            geoBlocked = webdriver.FirefoxOptions()
            geoBlocked.set_preference("geo.prompt.testing", True)
            geoBlocked.set_preference("geo.prompt.testing.allow", False)
            geoBlocked.set_preference("dom.push.enabled", False)
            self.driver = webdriver.Firefox(options=geoBlocked)
        else:
            self.driver = webdriver.Firefox()

        self.driver.get(self.url)

    def scrap_and_save_foundation_image(self):
        img = self.driver.find_elements(
            By.CSS_SELECTOR,
            'li.css-1xhaj19:nth-child(1) > button:nth-child(1) > svg:nth-child(1) > foreignObject:nth-child(1) > img:nth-child(1)'
        )
        if img != []:
            src = img[0].get_attribute('src')
            return src
        else:
            return ''

    def close_driver(self):
        self.driver.close()


