# project2_disaster_pipeline
## Overview
This is the second project in the [Udacity Data Scientist Nano Degree](https://www.udacity.com/course/data-scientist-nanodegree--nd025/ 'Course Overview'). It is a web app designed to be used in the event of a disaster in order to screen messages on social media and categorise them into one of 36 categories, which will help the agencies dealing with an emergency quickly identify which messages are appeals for aid and which agency should deal with them.

## Packages Used:
- pandas
- numpy
- nltk
- sklearn
- pickle

## Data
The disaster response data is supplied by [Figure Eight](https://www.figure-eight.com// 'Disaster Response from Figure Eight')

|Data Set|Type|Description|
|--------|----|-----------|
|disaster_messages|csv|Social media messages received during disasters|
|disaster_categories|csv|Categorisation of messages|

## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


## Screenshot
![screenshot](https://github.com/jennymcphail/project2_disaster_pipeline/blob/main/images/screenshot.JPG?raw=true)




