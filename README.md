# The FoundationFinder

It is reported that more than 100 million women in the U.S. have used makeup foundation in 2020 ([source](https://www.statista.com/statistics/275729/us-households-usage-of-foundation-and-concealer-make-up/#:~:text=Usage%20of%20foundation%20and%20concealer%20make%2Dup%20in%20the%20U.S.%202020&text=The%20data%20has%20been%20calculated,concealer%20make%2Dup%20in%202020)). A makeup foundation evens out people’s skin tones and covers flaws.  

When picking a foundation, a lot of people think about their skin types, because the same foundation can turn a person’s skin into a desert, but make another’s into a big oil field. And once you decided which foundation to buy, you need to find the right shade that matches your skin tone. There are almost 7,000 combination of foundation products and shades ([source](https://pudding.cool/2021/03/foundation-names/)), because each brand has a different set of shades. 

FoundationFinder saves time and money for the consumers by narrowing down the options from 7000 to 12. In addition, while the market value of makeup foundations was at $35.5 billion dollars in the U.S. in 2021 alone ([source](https://www.imarcgroup.com/face-makeup-market)), inefficiency remains in giving out free samples and handling returns to help consumers choose the right foundation and shade.

That's why I built the FoundationFinder, powered by machine learning.

## Here's how it looks like

<img src="https://github.com/ychu19/foundation/blob/main/img/UI_index.jpg" width=800px class="center">

First, users tell the app about their eye colors, hair colors, skin tones, and skin types. Like this:

<img src="https://github.com/ychu19/foundation/blob/main/img/UI_input.jpg" width=800px>

And then the app recommends the users **four** foundations, each with **three** shades, as shown here:

<img src="https://github.com/ychu19/foundation/blob/main/img/UI_predict.jpg" width=800px>

The app also has a link beneath each foundation that takes the users to the Sephora website for purchase.

## Here's how I built the app

### Data Collection

I used `Selenium` with `python` and scraped 120K reviews and information about 88 foundations from Sephora. A typical review looks like this:

<img src="https://github.com/ychu19/foundation/blob/main/img/example_review.jpg" width=800px>

Information scraped:
+ Whether the reviewers recommended the product or not
+ Ratings by each reviewer
+ Shades purchased by each reviewer
+ Texts in the review
+ Features of the reviewers (eye colors, hair colors, skin tones, skin types)
+ Date of the review
+ Information about each foundation (finish, coverage, skin types)

For an overview of the review data, please see the [data presentation](https://yuanningchu.web.illinois.edu/data_presentation.html).

### Data Cleaning and Feature Engineering

I cleaned the data, engineered the features with one-hot encoding, feature crossing, and keywords from the reviews with `scikit-learn` and `pandas`. During this process, I already had a picture in mind that *I will build one classifier for each foundation*. 

For each classifier, I have a binary label whether the reviewers recommended the product or not (1 for yes, 0 for no). And here is a list of the features I engineered from the data:
+ One-hot encoded features of the reviewers (eye colors, hair colors, skin tones, skin types)
+ Crossed feature between skin tones and skin types
+ Whether the reviewers liked the shade, the coverage, and the finish (extracted from texts in the review)
+ Month of the purchase (to capture seasonality)
+ Days since the first review of the product

### Model Training

After trying out a set of different classifiers with a few foundations, I trained an `XGBoost` classifier for each foundation to predict whether a person with specific characteristics will like the product or not. I tuned the hyperparameters with `GridSearchCV`. 

The models perform well on the validation set. I measured the performance of the models by AUC-ROC, area under the receiver operating characteristics curve. The AUC-ROC across all the models can be as high as 0.91 (see below), and when looking at predicted probability by true labels below, it shows that this classifier does a good job seperating the two classes apart.

<img src="https://github.com/ychu19/foundation/blob/main/img/xgb_auc_roc.jpeg" width=400px><img src="https://github.com/ychu19/foundation/blob/main/img/xgb_predictions_by_scores.jpeg" width=400px>

### Model Deployment

I deployed the models by saving all the model objects and one-hot encoders, and created a UI with `Flask` and `Bootstrap`.

## Backstage

<img src="https://github.com/ychu19/foundation/blob/scrapping/img/backstage.svg" width=600px>

### Candidate Generating Process

This app is still in its development stage, but I plan to have the app asking my users about their preferences for foundations, including finishes (natural, sheer, dewy, matte), coverages (natural, light, medium, full). With this input, the app narrows down the number of foundations that the users may be interested and generates an initial set of candidates. This step speeds up the time to make predictions, because the app does not have to go through all the predictors for each foundation to find the best matches.

<img src="https://github.com/ychu19/foundation/blob/scrapping/img/candidate_generating.svg" width=600px>

### Processing User Input

When a user provides information about their hair, eyes, skin tone, and skin type, the app processes the input (such as one-hot encoding) and feed the input into the predictors for all the initial candidates. 

### Making Predictions and Recommendations

After feeding the user input to the predictors, each predicting how likely the user will like the product, I rank all the predictors by their predicted probabilities and recommend the four foundations with the highest scores.

### Querying Best Possible Shades

With the four foundations the app recommends, I then query the data to look for most purchased shades among the satisfied reviewers who share similar characteristics with the user, and recommend the top three shades to the user.

