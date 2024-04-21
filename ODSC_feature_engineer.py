#!/usr/bin/env python
# coding: utf-8

# ## Feature Engineer
# This is the first in a series of three notebooks for the ODSC presentation 'Harnessing GPT Assistants for Superior Model Ensembles: A Beginner's Guide to AI STacked-Classifiers' ODSC East -- Jason Merwin

import time
import openai
import ipywidgets as widgets
from IPython.display import display
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import pandas as pd
from io import StringIO
import io
import json
import warnings

from config import OPENAI_API_KEY
warnings.filterwarnings('ignore', category=FutureWarning)

## Define Functions 
def delete_all_agents():
    ''' Deletes all exising Assistants associated with API key '''
    # Fetch the list of assistants
    my_assistants = client.beta.assistants.list(order="desc", limit=20)
    asst_ids = [asst.id for asst in my_assistants.data]
    print(f'Deleting {len(asst_ids)} assistants.')
    # Delete each assistant
    for asst_id in asst_ids:
        client.beta.assistants.delete(asst_id)
        print(f"Deleted assistant with ID: {asst_id}")
        time.sleep(1)
    print('Finished deleting all assistants')
    
def delete_all_assistant_files():
    ''' Deletes all exising files uploaded to OpenAI client using API key '''
    # generate a files object
    files_object = client.files.list()
    # get a list comprehension
    file_ids = [file.id for file in files_object.data]
    print(f'Deleting {len(file_ids)} files.')
    #delete them all
    for file_id in file_ids:
        client.files.delete(file_id)
        print(f"Deleted file with ID: {file_id}")
        time.sleep(1)
    print('Finished deleting all files')  

def upload_csv(file_name):
    """
    Sends a csv file to OpenAI and returns the file id
    file_name: string for name and ext of saved file. Example: "analyst_output.csv"
    return: file id
    """
    response = client.files.create(
        file=open(file_name, "rb"),
        purpose="assistants")
    print(response)
    file_id = response.id
    return file_id

def read_and_save_file(first_file_id, file_name):  
    """
    Reads the file contents from OpenAI file id and saves as csv.
    first_file_id: OpenAI file id
    file_name: string for name and ext of saved file. Example: "analyst_output.csv"
    """
    # its binary, so read it and then make it a file like object
    file_data = client.files.content(first_file_id)
    file_data_bytes = file_data.read()
    file_like_object = io.BytesIO(file_data_bytes)
    #now read as csv to create df
    returned_data = pd.read_csv(file_like_object)
    returned_data.to_csv(file_name, index=False)
    return returned_data

def files_from_messages(messages, asst_name):
    """
    Returns a csv data file from an OpenAI API message object.
    messages: OpenAI API messages object
    asst_name: string name of Assistant for use when saving file
    """
    first_thread_message = messages.data[0]  
    message_ids = first_thread_message.file_ids
    print(message_ids)
    # Loop through each file ID and save the file with a sequential name
    for i, file_id in enumerate(message_ids):
        file_name = f"{asst_name}_output_{i+1}.csv" 
        read_and_save_file(file_id, file_name)
        print(f'saved {file_name}')
        
def get_string_features(df):
    """
    Returns a list of column names in the DataFrame that are of string type.
    param df: pandas DataFrame
    return: List of column names that are strings
    """
    # Select columns of object dtype (commonly used for strings)
    string_columns = df.select_dtypes(include=['object']).columns.tolist()
    return string_columns  

def convert_strings_to_numbers(df, columns_to_convert):
    """
    Converts unique string values in specified columns to increasing numeric values.

    param df: pandas DataFrame
    param columns_to_convert: List of column names to be converted
    return: DataFrame with specified columns converted to numeric values
    """
    for column in columns_to_convert:
        # Ensure the column is in the DataFrame
        if column in df.columns:
            # Create a mapping from unique strings to numbers
            unique_strings = df[column].unique()
            string_to_number_mapping = {string: i for i, string in enumerate(unique_strings)}
            # Apply the mapping to the column
            df[column] = df[column].map(string_to_number_mapping)
        else:
            print(f"Column '{column}' not found in DataFrame.")
    return df

# upload data set and define target
training_df = pd.read_csv('apple_quality.csv')
target = 'Quality'
print(training_df.shape)
training_df.tail()

# Calculate the percentage of null values for each column
percent_missing = training_df.isnull().mean() * 100
print(percent_missing)

# check class distribution
normalized_distribution = training_df[f'{target}'].value_counts(normalize=True)
print(f'The percentage of class values in the data set are {normalized_distribution}')

# make copy of training data
train_df = training_df.copy()

print(train_df.shape)
print(train_df.tail())

# save the encoded training data
train_df.to_csv('pre_assistant_train.csv', index=False)
encoded_train = pd.read_csv('pre_assistant_train.csv')
encoded_train

# Instantiate the OpenAI client
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# clean up any existing assistants and files
delete_all_agents()   
delete_all_assistant_files()

prompt = f'''
You are a feature engineer who will create and test new features from a csv data set in your files. 
When the user asks you to perform your actions, use the csv file to read the data into a pandas dataframe.
The data set contains predictive features that will be used for a binary classifier.
Follow each of the steps listed below in your ACTIONS. The target variable is {target}. 

ACTIONS:

1. Read the file data into a pandas DataFrame. 
2. Check for missing values and impute the column mean for those missing values.
3. Numerically encode any categorical columns.
4. Create new feature interaction columns using the continuous, non categorical columns. For each unique pair of columns create a new column that is the result of multiplying one column by the other and a second column that is the result of dividing one column by the other. 
5. Add these new features to the original data set and run an extra trees random forest with 3,000 trees to predict the target variable 'Loan_Status'. 
6. Get the feature importances of all the features in the model and prepare the feature importance values as Table_1. This table should have one column for the features name and one for the importance value.
7. Now prepare a final training data set that contains the original continuous features, numerically encoded features, and the top 3 feature interactions based on their feature importance values. Prepare this final table as Table_2. Table 2 should have the 11 feature columns, the top 3 feature interactions, and the target {target}.
8. Prepare Table_1 and Table_2 for download by the user. 

DO NOT:
1. Do not return any images. 
2. Do not return any other tables besides Table_1 and Table_2
3. Do not use the target column {target} in the feature interactions.
4. Do not remove any of the original data set columns from the final data set Table_2.

'''

# send the csv file to the assistant purpose files
training_file_id = upload_csv('pre_assistant_train.csv')

# create the assistant and link to file
my_assistant = client.beta.assistants.create(
    instructions=prompt,
    name="feature_engineer",
    tools=[{"type": "code_interpreter"}],
    model="gpt-4-turbo-preview", 
    file_ids=[training_file_id]
)

# get the assistant file id
fileId = my_assistant.file_ids[0]
print(my_assistant)

# make the request to the assistant
message_string = f"Please execute your ACTIONS on the data stored in the csv file {fileId}. The Target variable is {target}"
print(message_string)

# Create a Thread
thread = client.beta.threads.create()
print('created thread')

# Add a Message to a Thread
message = client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content= message_string
)
print('added message to thread')

# Run the Assistant
run = client.beta.threads.runs.create(
    thread_id=thread.id,
    assistant_id=my_assistant.id
    #instructions="you can overwrite prompt instructions here"
)
print('running the client')

print('getting json response')
print(run.model_dump_json(indent=4))

# let an initial 5 minutes pass
time.sleep(360) 

# check for a response
while True:
    # Wait for 5 seconds
    time.sleep(60)  
    # Retrieve the run status
    run_status = client.beta.threads.runs.retrieve(
        thread_id=thread.id,
        run_id=run.id
    )
    print('One eternity later...')
    # If run is completed, get messages
    if run_status.status == 'completed':
        messages = client.beta.threads.messages.list(
            thread_id=thread.id
        )
        # Loop through messages and print content based on role
        for msg in messages.data:
            role = msg.role
            try:
                content = msg.content[0].text.value
                print(f"{role.capitalize()}: {content}")
            except AttributeError:
                # This will execute if .text does not exist
                print(f"{role.capitalize()}: [Non-text content, possibly an image or other file type]")
        break


# extract the file names from the response and retrieve the content
asst_name = 'feature_engineer'        
files_from_messages(messages, asst_name)

df1 = pd.read_csv('feature_engineer_output_1.csv')
display(df1)

import matplotlib.pyplot as plt

df2 = pd.read_csv('feature_engineer_output_2.csv')
df2 = df2.sort_values(by='Importance', ascending=False)
display(df2.head())

# Creating the plot
plt.figure(figsize=(10, 25))
plt.barh(df2['Feature'], df2['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.title('Feature Importance')
plt.grid(True, linestyle='--', alpha=0.6)  

# Show plot
plt.show()