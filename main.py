import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('news.csv')
x_news = df["text"]
y_news = df["label"]

def train_and_test(x_news, y_news):
    x_train, x_test, y_train, y_test = train_test_split(x_news, y_news, test_size=0.2, random_state=0)
    return x_train, x_test, y_train, y_test

def initialize_tfid(x_train, x_test):
    tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)
    
    tfidf_train=tfidf_vectorizer.fit_transform(x_train)
    tfidf_test=tfidf_vectorizer.transform(x_test)
    
    return tfidf_train, tfidf_test, tfidf_vectorizer

def initialize_passive_agressive_classifier(tfidf_train, tfidf_test, y_train, y_test):
    pac=PassiveAggressiveClassifier(max_iter=50)
    pac.fit(tfidf_train, y_train)
    
    y_pred=pac.predict(tfidf_test)
    score=accuracy_score(y_test, y_pred)
    print(score)

x_train, x_test, y_train, y_test = train_and_test(x_news, y_news)
tfidf_train, tfidf_test, tfidf_vectorizer = initialize_tfid(x_train, x_test)
initialize_passive_agressive_classifier(tfidf_train, tfidf_test, y_train, y_test)
    
    
    
    
