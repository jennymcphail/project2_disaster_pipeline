import sys
import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import FeatureUnion
import pickle

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def load_data(database_filepath):
    """ Loads data from sqlite database. Creates X (text) and Y (target variables).
    Input: database filepath
    Ouput: X, Y"""
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('disaster_response', engine)  
    X = df.pop('message')
    Y = df.drop(['id','original','genre'],axis=1)
    category_names = list(Y.columns)
    return X,Y, category_names
    pass


def tokenize(text):
    """Cleans (lower case, remove punctuation, remove stop words, lemmatize) and tokenizes (word tokens) text.
    Input: text
    Returns: tokens"""
    stop_words = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()
    # lower case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    # tokenize text
    tokens = word_tokenize(text)
    # lemmatize and remove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return tokens
    pass
    pass


def build_model():
    """Builds model using MultioutputClassifier with RandomForestClassifier, with Pipeline to apply count vectorization and tdif.
    Input: X, Y
    Output: model"""
    pipeline = Pipeline([('text_pipeline',Pipeline([('countvector',CountVectorizer(tokenizer=tokenize)) ,('tfidf', TfidfTransformer())]) ),('clf', MultiOutputClassifier(RandomForestClassifier())),])
    parameters = {'text_pipeline__countvector__min_df': [1, 5],
                'text_pipeline__tfidf__use_idf': [True, False],
                'clf__estimator__n_estimators': [10, 25],
                'clf__estimator__min_samples_split': [2, 4]}

    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv
    pass


def evaluate_model(model, X_test, Y_test, category_names):
    pred_test = model.predict(X_test)
    y_test = Y_test.values
    for i in range(36):
        print(category_names[i] + classification_report(y_test[:,i], pred_test[:,i]))
    pass


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()