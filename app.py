from flask import Flask, render_template, url_for, request, redirect, session
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
import xgboost as xgb
from predict import (
    predict_from_user_input,
    get_longest_dict,
    filter_shade,
    extracting_img_src,
    extracting_url,
    candidate_generation
)
import os
from datetime import datetime

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test.db'
app.config['SECRET_KEY'] = 'no_secret'
# db = SQLAlchemy(app)

# class Todo(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     content = db.Column(db.String(200), nullable=False)
#     date_created = db.Column(db.DateTime, default=datetime.utcnow)
#
#     def __repr__(self):
#         return '<Task %r>' % self.id


@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'GET':
        hairs = get_longest_dict(col='hair_color')
        # hairs = {key: val for key, val in hairs.items() if val != 'Red_hair'}
        eyes = get_longest_dict(col='eye_color')
        # tones = get_longest_dict(col='skin_tone')
        tones = ['Ebony', 'Deep', 'Dark', 'Olive', 'Medium', 'Fair', 'Light', 'Porcelain']
        # tones_img = [
        #     'https://www.sephora.com/productimages/sku/s2513869+sw.jpg',
        #     'https://www.sephora.com/productimages/sku/s2513885+sw.jpg',
        #     'https://www.sephora.com/productimages/sku/s2513919+sw.jpg',
        #     'https://www.sephora.com/productimages/sku/s2513984+sw.jpg',
        #     'https://www.sephora.com/productimages/sku/s2514123+sw.jpg',
        #     'https://www.sephora.com/productimages/sku/s2514149+sw.jpg',
        #     'https://www.sephora.com/productimages/sku/s2514180+sw.jpg',
        #     'https://www.sephora.com/productimages/sku/s2514255+sw.jpg'
        # ]
        # tones = zip(tones, tones_img)
        types = get_longest_dict(col='skin_type')
        return render_template('index.html', hairs=hairs, eyes=eyes, tones=tones, types=types)

    # else:
    #     features = request.form['hair']
    #     # return redirect('/predict.html', id='content')
    #     return redirect(url_for('.predict', features=features))

@app.route('/candidate/', methods=['GET'])
def candidate():
    if request.method == 'GET':
        eye = request.args.get('eye').replace('_eye', '')
        hair = request.args.get('hair').replace('_hair', '')
        tone = request.args.get('tone')
        type = request.args.get('type')
        finishes = ['natural', 'matte', 'radiant']
        coverages = ['medium', 'full', 'sheer', 'light']
        session['eye'] = eye
        session['hair'] = hair
        session['skin_tone'] = tone
        session['skin_type'] = type
        # selected_finish = request.form['finish']

        # return redirect(url_for('/predict'))
        return render_template(
            'candidate.html',
            finishes=finishes, coverages=coverages,
            type=type, tone=tone, hair=hair, eye=eye
        )


    # if request.method == 'POST':
    #     task_content = request.form['content']
    #     new_task = Todo(content=task_content)
    #
    #     try:
    #         db.session.add(new_task)
    #         db.session.commit()
    #         return redirect('/')
    #     except:
    #         return "There is an issue adding your task"
    # else:
    #     tasks = Todo.query.order_by(Todo.date_created).all()
    #     return render_template('index.html', tasks=tasks)

# @app.route('/delete/<int:id>')
# def delete(id):
#     task_to_delete = Todo.query.get_or_404(id)
#     try:
#         db.session.delete(task_to_delete)
#         db.session.commit()
#         return redirect('/') # homepage
#     except:
#         return "There was a problem deleting that task"
#
@app.route('/predict/', methods=['GET'])
def predict():
    candidate_dict = dict()
    candidate_dict['finish'] = request.args.get('finish')
    candidate_dict['coverage'] = request.args.get('coverage')
    candidate_dict['skin_type'] = session.get('skin_type')
    candidate_list = candidate_generation(candidate_dict)
    print(candidate_dict)
    features_dict = dict()
    features_dict['skin_tone'] = session.get('skin_tone')
    features_dict['skin_type'] = session.get('skin_type')
    features_dict['hair_color'] = session.get('hair')
    features_dict['eye_color'] = session.get('eye')
    # features = request.args.get('hair')
    features_dict['finish'] = 1
    features_dict['coverage'] = 1
    features_dict['shade_match'] = 1
    features_dict['gifted'] = 0
    features_dict['days_since_launch_scaled'] = 1
    features_dict['month_of_purchase'] = 3
    features_dict['skin_tone_cat'] = 0
    print(features_dict)
    # test_input = pd.read_json(
    #     "data_full_review_cleaned/bareMinerals_COMPLEXION_RESCUE™_Tinted_Moisturizer_with_Hyaluronic_Acid_and_Mineral_SPF_30.json",
    #     lines=True)
    # test_input = test_input.loc[20]
    # test_input = test_input[[
    #     "eye_color", "hair_color", "skin_type", "skin_tone", "finish", "coverage", "shade_match", "gifted",
    #     "days_since_launch_scaled", "month_of_purchase", "skin_tone_cat"
    # ]]
    # features = features.to_dict()
    scores = predict_from_user_input(features_dict, candidate_list)
    products = scores['brand_product'].to_dict().values()
    if all([i == '' for i in products]):
        return "Sorry, we don't have enough information to make recommendation for you."
    else:

        shade_data = pd.DataFrame()
        shade_data['brand_product'] = str()
        shade_data['shades'] = str()

        for idx in range(len(scores)):
            brand_product = scores.loc[idx, 'brand_product']
            proba = scores.loc[idx, 'scores']
            shades = filter_shade(input=features_dict, brand_product=brand_product)
            srcs = extracting_img_src(brand_product=brand_product)
            urls = extracting_url(brand_product=brand_product)
            cols = {
                'brand_product': brand_product.replace('_', ' '),
                'scores': round(proba*100, 1),
                'shades': shades,
                'srcs': srcs,
                'urls': urls
            }
            temp_data = pd.DataFrame([cols])
            shade_data = pd.concat([shade_data, temp_data], axis=0, ignore_index=True)

        products = shade_data['brand_product']
        shades = shade_data['shades']
        srcs = shade_data['srcs']
        urls = shade_data['urls']
        scores = shade_data['scores']

        products_shades = zip(products, shades, srcs, urls, scores)
        print(scores)
        return render_template('predict.html', products_shades=products_shades)

    # task = Todo.query.get_or_404(id)

    # if request.method == 'POST':
    #     # pass
    #     task.content = request.form['content']
    #
    #     try:
    #         db.session.commit()
    #         return redirect('/') # homepage
    #
    #     except:
    #         return "There was an issue updating that task"
    #
    # else:
    #     return render_template('update.html', task=task)

if __name__ == "__main__":
    app.run(debug=True)