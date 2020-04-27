# Disaster Response Pipeline Project

## Installations Required

This project is donw using Python 3.7
- pandas
- numpy
- sqlalchemy
- nltk
- sklearn
- re
- joblib
- flask

## Summary
In this project we have a data set containing real messages that were sent during disaster events.We created a machine learning pipeline to categorize these events so that we can send the messages to an appropriate disaster relief agency.
We buit an webapp where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data.

## Components
- app
    - run.py => Contains the code for visualisation and initialize the web app
    - templates => html files
- data
    - Contains csv files of data provided by figure eight => disaster_categories.csv disaster_messages.csv
    - process_data.py => Takes csv files as input then cleans the data and create SQL database.
- model
    - train_classifier.py
        - loads the data from sql database
        - Build text processing and machine learning pipeline
        - Train and tune model using GridSearchCV
        - Export the final model as pickle file

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://127.0.0.1:3001/
