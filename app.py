from flask import Flask, render_template, request
import pandas as pd
from thyroid.predictor import ModelResolver
from thyroid.utils import load_object 
from thyroid.logger import logging
from thyroid.exception import ThyroidException
import os, sys

app = Flask(__name__)

# path for latest transformer , model and target encoder
model_resolver = ModelResolver()
transformer_path = model_resolver.get_latest_transformer_path()
logging.info(f"Transformer path: {transformer_path}")
model_path = model_resolver.get_latest_model_path()
logging.info(f"model_path: {model_path} ")
target_encoder_path = model_resolver.get_latest_target_encoder_path()


# loading the transformer, model and target encoder
transformer = load_object(file_path=transformer_path)
model = load_object(file_path=model_path)
target_encoder = load_object(file_path=target_encoder_path)

@app.route("/", methods=['GET','POST'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST','GET'])
def predict():
    try:
        # Extract the form data
        age = int(request.form.get('age'))
        sex = request.form.get('sex')
        on_thyroxine = request.form.get('on_thyroxine')
        query_on_thyroxine = request.form.get('query_on_thyroxine')
        on_antithyroid_medication = request.form.get('on_antithyroid_medication')
        sick = request.form.get('sick')
        pregnant = request.form.get('pregnant')
        thyroid_surgery = request.form.get('thyroid_surgery')
        I131_treatment = request.form.get('I131_treatment')
        query_hypothyroid = request.form.get('query_hypothyroid')
        query_hyperthyroid = request.form.get('query_hyperthyroid')
        lithium = request.form.get('lithium')
        goitre = request.form.get('goitre')
        tumor = request.form.get('tumor')
        psych = request.form.get('psych')
        hypopituitary = request.form.get('hypopituitary')
        T3 = float(request.form.get('T3'))
        TT4 = float(request.form.get('TT4'))
        T4U = float(request.form.get('T4U'))
        FTI = float(request.form.get('FTI'))
        referral_source = request.form.get('referral_source')

        # Create a Pandas DataFrame from the form data
        data = pd.DataFrame({
            'age': [age],
            'sex': [sex],
            'on_thyroxine': [on_thyroxine],
            'query_on_thyroxine': [query_on_thyroxine],
            'on_antithyroid_medication': [on_antithyroid_medication],
            'sick': [sick],
            'pregnant': [pregnant],
            'thyroid_surgery': [thyroid_surgery],
            'I131_treatment': [I131_treatment],
            'query_hypothyroid': [query_hypothyroid],
            "query_hyperthyroid":[query_hyperthyroid],
            'lithium': [lithium],
            'goitre': [goitre],
            'tumor': [tumor],
            'psych': [psych],
            'hypopituitary': [hypopituitary],
            'T3': [T3],
            'TT4': [TT4],
            'T4U': [T4U],
            'FTI': [FTI],
            'referral_source': [referral_source]
        })
        if request.method == 'POST':
            input_arr = transformer.transform(data)
            y_pred = model.predict(input_arr)
            prediction = target_encoder.inverse_transform(y_pred)
            return render_template('index.html', prediction=prediction)
        else:
            return render_template('index.html', prediction="")

    
    except Exception as e:
        raise ThyroidException(e, sys)

if __name__=="__main__":
    app.run(host="0.0.0.0", debug=True)

