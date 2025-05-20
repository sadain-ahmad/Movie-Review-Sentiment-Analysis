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

df = pd.read_csv('D:\\My Journey\\Machine Learning\\Python Pandas\\Pandas Lectures\\Dataset\\IMDB Dataset.csv')

df.drop_duplicates(inplace=True)
x_train, x_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], test_size=0.1, random_state=42)
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.fit_transform(y_test)

# expanding abreviations
def expanding_abrevations(text):
    text = contractions.fix(text)
    text = re.sub(r"/bb4/b", "before", text)
    text = re.sub(r"/bl8r/b", "later", text)
    text = re.sub(r"/bgr8/b", "great", text)
    text = re.sub(r"/bb4n/b", "bye for now", text)
    text = re.sub(r"/byw/b", "you are welcome", text)
    text = re.sub(r"/bbbl/b", "be back later", text)
    text = re.sub(r"/bgtg/b", "got to go", text)
    text = re.sub(r"/bjk/b", "just kidding", text)
    text = re.sub(r"/bnp/b", "no problem", text)
    text = re.sub(r"/blmao/b", "laugh", text)
    text = re.sub(r"/bcuz/b", "because", text)
    text = re.sub(r"/blol/b", "laughing out loud", text)
    text = re.sub(r"/bomg/b", "oh my god", text)

    text = re.sub(r"/bbtw/b", "by the way", text)
    text = re.sub(r"/bidk/b", "I do not know", text)
    text = re.sub(r"/btbh/b", "to be honest", text)
    text = re.sub(r"/bthx/b", "thanks", text)
    text = re.sub(r"/bgotta/b", "got to", text)
    text = re.sub(r"/bwanna/b", "want to", text)
    text = re.sub(r"/bgonna/b", "going to", text)
    text = re.sub(r"/bgimme/b", "give me", text)
    text = re.sub(r"lemme/b", "let me", text)
    text = re.sub(r"/bb/c/b", "because", text)
    text = re.sub(r"/boz/b", "oswald", text)
    text = re.sub(r"/bomw/b", "on my way", text)

    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"who's", "who is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"there's", "there is", text)
    text = re.sub(r"here's", "here is", text)
    text = re.sub(r"peach's", "peach is", text)
    
    text = re.sub(r"they're", "they are", text)
    text = re.sub(r"who're", "who are", text)
    text = re.sub(r"what're", "what are", text)
    text = re.sub(r"where're", "where are", text)
    text = re.sub(r"how're", "how are", text)
    text = re.sub(r"there're", "there are", text)
    text = re.sub(r"here're", "here are", text)
    text = re.sub(r"they're", "they are", text)
    text = re.sub(r"we're", "we are", text)
    text = re.sub(r"you're", "you are", text)
    
    text = re.sub(r"i'm", "I am", text)
    
    text = re.sub(r"he'll", "he will", text)
    text = re.sub(r"she'll", "she will", text)
    text = re.sub(r"it'll", "it will", text)
    text = re.sub(r"they'll", "they will", text)
    text = re.sub(r"we'll", "we will", text)
    text = re.sub(r"you'll", "you will", text)
    text = re.sub(r"who'll", "who will", text)
    text = re.sub(r"what'll", "what will", text)
    text = re.sub(r"i'll", "I will", text)
    
    text = re.sub(r"i've", "I have", text)
    text = re.sub(r"you've", "you have", text)
    text = re.sub(r"he've", "he has", text)
    text = re.sub(r"she've", "she has", text)
    text = re.sub(r"we've", "we have", text)
    text = re.sub(r"they've", "they have", text)
    text = re.sub(r"who've", "who has", text)
    text = re.sub(r"what've", "what has", text)
    text = re.sub(r"here've", "here have", text)
    text = re.sub(r"there've", "there have", text)
    text = re.sub(r"where've", "where have", text)
    text = re.sub(r"how've", "how have", text)

    text = re.sub(r"i'd", "I would", text)
    text = re.sub(r"you'd", "you would", text)
    text = re.sub(r"he'd", "he would", text)
    text = re.sub(r"she'd", "she would", text)
    text = re.sub(r"we'd", "we would", text)
    text = re.sub(r"they'd", "they would", text)
    text = re.sub(r"who'd", "who would", text)
    text = re.sub(r"what'd", "what would", text)
    text = re.sub(r"here'd", "here would", text)
    text = re.sub(r"there'd", "there would", text)
    text = re.sub(r"where'd", "where would", text)
    text = re.sub(r"how'd", "how would", text)
    text = re.sub(r"it'd", "it would", text)

    text = re.sub(r"can't", "can not", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"is't", "is not", text)
    text = re.sub(r"aren't", "are not", text)
    text = re.sub(r"wasn't", "was not", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"don't", "do not", text)
    text = re.sub(r"doesn't", "does not", text)
    text = re.sub(r"hasn't", "has not", text)
    text = re.sub(r"haven't", "have not", text)
    text = re.sub(r"hadn't", "had not", text)
    text = re.sub(r"wasn't", "was not", text)
    text = re.sub(r"weren't", "were not", text)
    text = re.sub(r"shouldn't", "should not", text)
    text = re.sub(r"couldn't", "could not", text)
    text = re.sub(r"wouldn't", "would not", text)
    text = re.sub(r"mightn't", "might not", text)
    text = re.sub(r"mustn't", "must not", text)

    text = re.sub(r"let's", "let us", text)
    text = re.sub(r"y'all", "you all", text)
    text = re.sub(r"ain't", "am not", text)
    return text

# tokenization
def tokenize(text):
    text =  word_tokenize(text, language='english', preserve_line=True)
    return text

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    return [word for word in text if word not in stop_words]

def stemming(text):
    porter = PorterStemmer()
    return [porter.stem(word) for word in text]

def clean(x):
    x = x.str.lower()
    x = x.str.strip()
    x = x.str.replace('<.*?>', '', regex=True)
    x = x.str.replace(r'http:\S+|www.\S+|https:\S+', '', regex=True)
    x = x.str.replace('[^\w\s]', '', regex=True)
    
    return x

# create a pipeline
pipe = Pipeline(
    steps=[
        ('clean', FunctionTransformer(clean)),
        ('vectorization', CountVectorizer(max_features=2000)),
        ('model', MultinomialNB())
    ], 
    verbose=True
)
if __name__ == '__main__':
    pipe.fit(x_train, y_train)

    y_pred = pipe.predict(x_test)
    print("Accuracy: ", accuracy_score(y_test, y_pred))
    print('confusion matrix : \n', confusion_matrix(y_test, y_pred))
