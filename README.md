# 🎬 Movie Review Sentiment Analysis

This project is a Natural Language Processing (NLP) solution that classifies movie reviews as **positive** or **negative**. It uses a **Multinomial Naive Bayes** model and has been deployed using **Flask** for real-time sentiment prediction via a web interface.

## 🔍 Project Overview

- **Objective**: Automatically determine the sentiment (positive/negative) of a movie review.
- **Model Used**: Multinomial Naive Bayes
- **Accuracy**: 83.88% using 10-fold cross-validation
- **Deployment**: Flask-based web application

## 🧠 Features

- End-to-end NLP pipeline:
  - HTML tag removal
  - URL removal
  - Abbreviation expansion
  - Lowercasing
  - Tokenization
  - Stop word removal
  - Stemming
- Text vectorization using `CountVectorizer`
- Sentiment classification using `MultinomialNB`
- Real-time prediction via a simple and user-friendly Flask web interface

## ⚙️ Tech Stack

- **Language**: Python
- **Libraries**: 
  - `scikit-learn`
  - `NLTK`
  - `Flask`
  - `re`, `string`, `joblib`, `pandas`, `numpy`

## 🚀 How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/movie-review-sentiment-analysis.git
cd movie-review-sentiment-analysis
```

### 2. Create and Activate a Virtual Environment (optional)

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Flask App

```bash
python app.py
```

Open your browser and navigate to http://127.0.0.1:5000/ to access the app.


## 📊 Model Evaluation
**Cross-Validation Score**: 83.88%

Evaluated using cross_val_score with 10-fold CV

**Model**: Multinomial Naive Bayes

## 📫 Contact
**Sadain Ahmad**
Feel free to connect on **LinkedIn**: www.linkedin.com/in/sadain-ahmad-91b679309
Or drop a message at [sadainahmad794@gmsil.com]