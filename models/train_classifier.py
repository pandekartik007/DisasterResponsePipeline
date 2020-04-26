import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score,make_scorer
from sklearn.model_selection import GridSearchCV
import joblib

def load_data(database_filepath):
    '''
    Load and merge dataset from the sql database
    Args
        database_filepath: String, filepath to access the sql database
    Return
        X: Pandas DataFrame, message data
        Y: Pandas DataFrame, labels
        category_names: String, names of all the columns available
    '''
    # load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql("SELECT * FROM Messages", engine)
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis = 1)
    category_names = Y.columns.tolist()
    return X,Y,category_names


def tokenize(text):
    '''
    tokenize text: remove punctuations and stop words then lemmatize
    Args
        text: String, message to be tokenized
    Return  
        tokens: List, list of clean tokens 
    '''
    #All stop words in english
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    #remove punctuation and lower case the text
    text = re.sub(r"[^a-zA-Z0-9]"," ", text.lower())
    #tokenizing
    tokens = word_tokenize(text)
    #removing stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return tokens


def build_model():
    '''
    construction of pipeline
    Return : GridSearchCV model based on the parameters
    The best parameters after testing from this model are
    {'clf__estimator__max_features': 'auto', 'clf__estimator__min_samples_split': 4, 
    'clf__estimator__n_estimators': 25, 'tfidf__use_idf': 'False', 
    'vect__max_df': 0.8}
    '''
    pipeline = Pipeline([
    ('vect',CountVectorizer(tokenizer=tokenize)),
    ('tfidf',TfidfTransformer()),
    ('clf',MultiOutputClassifier(RandomForestClassifier()))
    ])
    #parameters selected
    parameters = {
    'clf__estimator__n_estimators': [10,25],
    'clf__estimator__max_features': ['auto','log2'],
    'clf__estimator__min_samples_split':[2,4],
    'tfidf__use_idf':['True','False'],
    'vect__max_df':[0.8,1.0]
    }
    # create grid search object, n_jobs=-1 => using all available cores of cpu
    cv = GridSearchCV(pipeline, param_grid = parameters,cv=3,verbose =10,n_jobs=-1)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Classification results displaying f1_score,recall and precision for each category
    Args
        model: GridSearchCV model with the best parameters
        X_test, Y_test: test sets for our dataset 
    '''
    y_pred = model.predict(X_test)
    for i,col in enumerate(Y_test):
        print(col)
        #prints out f1_score,precision,recall for each category
        print(classification_report(Y_test[col],y_pred[:,i]))


def save_model(model, model_filepath):
    '''
    Save model as pickle file
    Args
        model: selected GridSearchCV model
        model_filepath: name and location of the pickle file to be saved
    '''
    joblib.dump(model, model_filepath)


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