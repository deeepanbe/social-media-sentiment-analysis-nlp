#!/usr/bin/env python3
"""
sentiment_analyzer.py
Social Media Sentiment Analysis using NLP
Author: Deepanraj A  
"""

import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import re

def clean_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    return text.strip()

def analyze_sentiment_vader(df):
    analyzer = SentimentIntensityAnalyzer()
    df['vader_score'] = df['text'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
    df['sentiment'] = df['vader_score'].apply(lambda x: 'positive' if x > 0.05 else ('negative' if x < -0.05 else 'neutral'))
    return df

def train_ml_model(X_train, y_train):
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_vec, y_train)
    return model, vectorizer

def main():
    df = pd.read_csv('data/tweets.csv')
    df['cleaned_text'] = df['text'].apply(clean_text)
    df = analyze_sentiment_vader(df)
    
    print(f"Analyzed {len(df)} tweets")
    print(df['sentiment'].value_counts())
    
    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned_text'], df['sentiment'], test_size=0.2, random_state=42
    )
    
    model, vectorizer = train_ml_model(X_train, y_train)
    X_test_vec = vectorizer.transform(X_test)
    accuracy = model.score(X_test_vec, y_test)
    
    print(f"Model Accuracy: {accuracy:.4f}")

if __name__ == '__main__':
    main()
