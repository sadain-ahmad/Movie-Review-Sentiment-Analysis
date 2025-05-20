from flask import Flask, render_template, request, redirect, url_for, flash
import numpy as np
import pandas as pd
import contractions
import re
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import FunctionTransformer
import pickle
from model import (clean, expanding_abrevations, remove_stopwords, stemming, tokenize)


with open('D:\\My Journey\\Machine Learning\\Projects\\Sentiment Analysis\\model.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def projects():
    return render_template('projects.html')

@app.route('/model.html')
def model_page():
    return render_template('sentiment_analyzer.html')



@app.route('/get_input', methods=['post'])
def get_input():
    # Get the input from the request
    input_text = request.form.get('review')
    # Preprocess the input text
    input_text = pd.Series(input_text)
    print(model.predict(input_text))

    y_pred = model.predict(input_text)
    # Convert the prediction to a integer
    y_pred = int(y_pred[0])
    # Render the result template with the prediction

    if y_pred == 1:
        return render_template('sentiment_analyzer.html', message = 1)
    elif y_pred == 0:
        return render_template('sentiment_analyzer.html', message = 0)
    else:
        return render_template('sentiment_analyzer.html', message="Neutral")



app.run(debug=True)