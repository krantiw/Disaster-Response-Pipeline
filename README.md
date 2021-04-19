# Disaster-Response-Pipeline

GitHub repository link of my project: https://github.com/krantiw/Disaster-Response-Pipeline


The goal of this project is to apply the data engineering skills learned in the course to analyze disaster data from [Figure Eight](https://www.figure-eight.com/) to build a model for an API that classifies disaster messages. The project is divided in three sections:
* **Data Processing**: build an ETL (Extract, Transform, and Load) Pipeline to extract data from the given dataset, clean the data, and then store it in a SQLite database
* **Machine Learning Pipeline**: split the data into a training set and a test set. Then, create a machine learning pipeline that uses NLTK, as well as scikit-learn's Pipeline and GridSearchCV to output a final model that predicts a message classifications for the 36 categories (multi-output classification)
* **Web development** develop a web application to show classify messages in real time

## Software and Libraries

This project uses Python 3.7.2 and the following libraries:
* [NumPy](http://www.numpy.org/)
* [Pandas](http://pandas.pydata.org)
* [nltk](https://www.nltk.org/)
* [scikit-learn](http://scikit-learn.org/stable/)
* [sqlalchemy](https://www.sqlalchemy.org/)

## Data

The dataset is provided by [Figure Eight](https://www.figure-eight.com/dataset/combined-disaster-response-data/) is basically composed by:
* **disaster_categories.csv**: Categories of the messages
* **disaster_messages.csv**: Multilingual disaster response messages

### Important Files
**app/templates/***: templates/html files for web app

**data/process_data.py**: Extract Train Load (ETL) pipeline used for data cleaning, feature extraction, and storing data in a SQLite database

**models/train_classifier.py**: A machine learning pipeline that loads data, trains a model, and saves the trained model as a .pkl file for later use

**run.py**: This file can be used to launch the Flask web app used to classify disaster messages

## Running the code

Run the following command in the app's directory to run your web app. python run.py
Go to http://0.0.0.0:3001/

<pre>
.
├── app     
│   ├── run.py                           # Flask file that runs app
│   └── templates   
│       ├── go.html                      # Classification result page of web app
│       └── master.html                   # Main page of web app
├── models
│   └── train_classifier.py              # Train ML model       
├── data                   
│   ├── disaster_categories.csv          # Dataset including all the categories  
│   ├── disaster_messages.csv            # Dataset including all the messages
│   └── process_data.py                  # Data cleaning       
└── README.md
</pre>


## Instructions to run each file:

1. Run the following commands in the project's directory. This will clean the files,set up the database, train model and save the model.
 - To run ETL pipeline to clean data and store the processed data in the database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/disaster_response_db.db`
     
     Screenshot of process_data.py
     
 


       
  - To run the ML pipeline that loads data from DB, trains classifier and saves the classifier as a pickle file
        `python models/train_classifier.py data/disaster_response_db.db models/classifier.pkl`
    
     Screenshot of train_classifier.py with precision, recall etc. for each category
    
        
 2.- To run your web app in the app's directory
      `python run.py` 
      Go to http://0.0.0.0:3001/
      It will classify the given text
     ![sample_input](https://user-images.githubusercontent.com/70027063/115120055-066d4780-9fc9-11eb-8f75-4a8b1bc4b37c.png)



        
   After clicking Classify Message, we can see the categories which the message belongs to highlighted in green
   ![sample_output](https://user-images.githubusercontent.com/70027063/115119694-51865b00-9fc7-11eb-9003-4db545e71b54.png)


The main page shows some graphs about training dataset, provided by Figure Eight

![main_page](https://user-images.githubusercontent.com/70027063/115119921-6c0d0400-9fc8-11eb-8af7-3bd1b924ee13.png)
