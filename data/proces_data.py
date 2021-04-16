"""
Preprocessing of Data
Project: Disaster Response Pipeline (Udacity - Data Science Nanodegree)

Sample Script Syntax:

> python process_data.py <path to messages csv file> <path to categories csv file> <path to sqllite  destination db>

Sample Script Execution:
> python process_data.py disaster_messages.csv disaster_categories.csv disaster_response.db

Arguments Description:
    1) Path to the CSV file containing messages (e.g. disaster_messages.csv)
    2) Path to the CSV file containing categories (e.g. disaster_categories.csv)
    3) Path to SQLite destination database (e.g. disaster_response.db)
"""

#Import all the neccessary libraries
import pandas as pd
from sqlalchemy import create_engine
import sys

def load_data(messages_filepath, categories_filepath):
    """   
      Load messages data with categories
     Arguments:
      messages_filepath -> Path to the CSV file containing messages
      categories_filepath -> Path to the CSV file containing categories
     Output:
         df -> Combined data containing messages and categories

    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on="id")
    return df


def clean_data(df):
    """
    Clean the category dataset

    Arguments:
    df -> Combined data containing messages and categories
    Output:
         df -> Combined data containing messages and cleaned categories

    """
    #Split categories into separate category columns.
    data_categories= df["categories"].str.split(";", expand= True)
    
    #Rename columns of categories with new column names.
    new_header = data_categories.iloc[0]
    new_header = new_header.str.split("-")
    data_categories.columns = [row[0] for row in new_header]
    data_categories.head()
    
    for column in data_categories:
    
        data_categories[column] = [value[1] for value in data_categories[column].str.split("-") ]
    
    #convert the columns data to numeric
    data_categories[column] = pd.to_numeric(data_categories[column])
    data_categories.head()
    
    df.drop("categories", axis=1, inplace= True)
    df = pd.concat([df, data_categories], axis=1)
    df = df.drop_duplicates()
    
    

    
def save_data(df, database_filename):
    '''
    Save df into sqlite db
    Input:
        df: cleaned dataset
        database_filename: database name
    Output: 
        A SQLite database
    '''
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('DisasterMessages', engine, index=False)
    
print(sys.argv)

def main():
    if len(sys.argv) == 4:
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]
        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)
        print('Cleaning data...')
        df = clean_data(df)        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)        
        print('Cleaned data saved to database!')    
    else:
        print('Example: python process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db')


if __name__ == '__main__':
    main()