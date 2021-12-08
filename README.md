# project2_disaster_pipeline
## Overview
This is the second project in the Udacity Data Scientist Nano Degree. It is a web app designed to be used in the event of a disaster in order to screen messages on social media and categorise them into one of 36 categories, which will help the agencies dealing with an emergency quickly identify which messages are appeals for aid and which agency should deal with them.

## Packages Used:
- pandas
- numpy
- nltk
- sklearn
- pickle

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

![screenshot](https://github.com/jennymcphail/project2_disaster_pipeline/images/screenshot.JPG?raw=true)
