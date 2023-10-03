from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler # For our preprocessing.pkl

from src.pipeline.predict_pipeline import CustomData, PredictionPipeline

# Creating a flask app which is a WSGI Application that communicates a webserver and web application we create
application=Flask(__name__)

app = application

# Decorator Route -> Decoartor extends the behavior of a function without changing it
@app.route('/') # (rule, options)
def index():
    return render_template("index.html") # Will search for the "template" folder

@app.route("/predict", methods=["POST", "GET"])
def predict():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            gender = request.form.get('gender'),
            race_ethnicity = request.form.get('ethnicity'),
            parental_level_of_education = request.form.get('parental_level_of_education'),
            lunch = request.form.get('lunch'),
            test_preparation_courses = request.form.get('test_preparation_course'),
            reading_score = float(request.form.get('reading_score')),
            writing_score = float(request.form.get('writing_score'))
        )
        # Function that converts data to dataframe in predict_pipeline
        prediction_df = data.get_data_as_data_frame()
        predict_pipeline = PredictionPipeline()
        results = predict_pipeline.predict(prediction_df)
        return render_template('home.html', results = results[0])




if __name__ == "__main__":
    app.run(host="0.0.0.0") # debug=True will change the website without restarting, just save