#!/usr/bin/env python
# coding: utf-8

# ## Base Model Swarm
# This is the second in a series of three notebooks for the ODSC presentation 'Harnessing GPT Assistants for Superior Model Ensembles: A Beginner's Guide to AI STacked-Classifiers' ODSC East -- Jason Merwin

import openai
import time
import ipywidgets as widgets
from IPython.display import display
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from io import StringIO
import io
import json
import warnings

from config import OPENAI_API_KEY

warnings.filterwarnings('ignore', category=FutureWarning)

# define functions

def delete_all_agents():
    ''' Deletes all exising Assistants '''
    # Fetch the list of assistants
    my_assistants = client.beta.assistants.list(order="desc", limit=20)
    asst_ids = [asst.id for asst in my_assistants.data]
    print(f'Deleting {len(asst_ids)} assistants.')
    # Delete each assistant
    for asst_id in asst_ids:
        client.beta.assistants.delete(asst_id)
        print(f"Deleted assistant with ID: {asst_id}")
    print('Finished deleting all assistants')
    
def delete_all_assistant_files():
    ''' Deletes all exising files uploaded to client using API key '''
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
    response = client.files.create(
        file=open(file_name, "rb"),
        purpose="assistants")
    print(response)
    file_id = response.id
    return file_id

def spin_up(target, base_instructions, file_id):
    # create assistant
    my_assistant = client.beta.assistants.create(
        instructions=base_instructions,
        name="agent",
        tools=[{"type": "code_interpreter"}],
        model="gpt-4-turbo-preview", #"gpt-4-1106-preview", # "gpt-4", # "gpt-3.5-turbo-1106", "gpt-4-turbo-preview"
        file_ids=file_id)
    message_string = "Please execute your ACTIONS on the csv file, the target field is " + target
    # Create a Thread
    thread = client.beta.threads.create()
    # Add a Message to a Thread
    message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content= message_string)
    # Run the Assistant
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=my_assistant.id)
    return my_assistant, thread, run 
    print('Finished creating Assistants')
    
def catch_response(assistant, thread, run):
    # Retrieve the run status
    run_status = client.beta.threads.runs.retrieve(
        thread_id=thread.id,
        run_id=run.id)
    print('########################')
    print('Checking for response...')
    # Handle None response
    if run_status is None:
        print("No response yet")
        return None, None  # Return a tuple of None values to match the expected return type
    # Handle non-completed response
    if run_status.status != 'completed':
        print("Response status is not 'completed'")
        return None, None
    # Handle completed response
    if run_status.status == 'completed':
        messages = client.beta.threads.messages.list(
            thread_id=thread.id)
        contents = []  # Initialize an empty list to store contents
        # Loop through messages and process content based on role
        for msg in messages.data:
            role = msg.role
            try:
                content = msg.content[0].text.value
                print(f"{role.capitalize()}: {content}")
                contents.append(content)  # Append content to the list
            except AttributeError:
                # This will execute if .text does not exist
                print(f"{role.capitalize()}: [Non-text content, possibly an image or other file type]")
        return messages, contents  # Return messages and a list of contents
    else:
        print('Unable to retrieve message')
        return None, None

def create_dataframes_from_messages(messages, client):
    loop_dfs = []
    # Accessing the first ThreadMessage
    first_thread_message = messages.data[0]  
    message_ids = first_thread_message.file_ids
    # Loop through each file ID and create a DataFrame
    for file_id in message_ids:
        # Read the file content
        file_data = client.files.content(file_id)
        file_data_bytes = file_data.read()
        file_like_object = io.BytesIO(file_data_bytes)
        # Create a DataFrame from the file-like object and append
        df = pd.read_csv(file_like_object)
        loop_dfs.append(df)
    return loop_dfs    


# Instantiate the OpenAI client
client = openai.OpenAI(api_key=OPENAI_API_KEY)

#use the feature engineer output
target = 'Quality'
encoded_train = pd.read_csv('feature_engineer_output_1.csv')

#optional: use the original dataset instead
#encoded_train = pd.read_csv('pre_assistant_train.csv')

#add a row id 
encoded_train = encoded_train.reset_index()
encoded_train = encoded_train.rename(columns={'index': 'row_id'})
encoded_train

# first make sure any existing bots and files are cleaned up
delete_all_agents()   
delete_all_assistant_files()

#reserve 20% of training data to be used as "inference" data
train_set, val_set = train_test_split(encoded_train, test_size=0.2)

#save the files
train_set.to_csv('encoded_train.csv', index=False)
val_set.to_csv('encoded_val.csv', index=False)

#define the model types here by description
model_types = ['Logistic_Regression', 'DecisionTreeClassifier', 'KNeighborsClassifier', 'Random_Forest', 'Extra_Trees_Random_Forest', 'Support Vector Machine']

train_id = upload_csv(f'encoded_train.csv')
val_id = upload_csv(f'encoded_val.csv')
file_ids = [train_id, val_id]

agents = []

for i in model_types:
    print(f'Creating {i} assistant')
    
    #assign loop version of models and file names
    model = i
    print('provided these files')
    print(file_ids)
    instructions = instructions = f'''
    You are a data scientist who will build and test a predictive model with data from the provided csv file.
    This model will be base model for a stacked model ensemble, thus the predictions on the training data will be used as input for a meta model. 
    When the user asks you to perform your ACTIONS, carry out the described ACTIONS on the provided files.
    The target variable is '{target}'.
    There is an id column to be maintained, unaltered and returned in the output called "row_id". This column should be excluded when training the model.

    ACTIONS:

    1.The data has been prepared for training a {model} classification model to predict the target variable '{target}'.
    2.Split the training data in the file {train_id} into 5 K folds for cross-validation. Each fold should serve once as a validation set while the remaining folds serve as training sets.
    3.Train a {model} classification model using default hyper-parameter values on each training set derived from the K folds, ensuring the target variable is '{target}'.
    4.For each fold, use the trained {model} to predict the '{target}' on its corresponding validation set. Ensure the predictions are probabilities.
    5.Compile the out-of-fold predictions into a single dataset. This dataset should include the 'row_id' from the testing set and the predicted probabilities. Name the columns as follows: 'row_id' and '{model[:4]}_prob'.
    6.Save this compiled dataset as a CSV file named 'out_of_fold_predictions.csv' and prepare it for the user to download. This file will be used for training the meta-model.
    7.Now use the trained models to score the validation data in the file {val_id} containing the same target column '{target}'. Average their scores for each row in the validation data and compile the results in the same way as before and prepare it for the user to download as a CSV file names 'valiation_predicitons.csv'
    8.Both tables should contain 2 columns: row_id and '{model[:4]}_prob'.
    9.Please only respond once, with both tables once they are ready for download.
    
    DO NOT:
    1. Do not return any images.
    2. Do not return any other tables besides the tables 'out_of_fold_predictions.csv' and 'valiation_predicitons.csv'
    3. Do not include row_id as a feature in the training of the model.
    4. Do not respond before both tables are ready for download.

    '''  

    # spin up for each model type and store return object
    assistant, thread, run = spin_up(f'{target}', instructions, file_ids) 
    agents.append((assistant, thread, run, model))  
    print()
    time.sleep(5)

# run a loop to catch the Agent responses
time.sleep(360) 

agent_responses = []
for assistant, thread, run, model, in agents:
    messages, content = catch_response(assistant, thread, run) 
    agent_responses.append((messages, content, model, assistant))
    time.sleep(10) 


#extract dataframes and compile
import io
import pandas as pd

def create_dataframes_from_messages(messages, client):
    loop_dfs = []

    # Check if messages is None or messages.data is empty
    if messages is None or not messages.data:
        print("No messages data found.")
        return loop_dfs

    first_thread_message = messages.data[0]  # Accessing the first ThreadMessage
    message_ids = first_thread_message.file_ids

    # Loop through each file ID and create a DataFrame
    for file_id in message_ids:
        # Read the file content
        file_data = client.files.content(file_id)

        # Check if file_data is None
        if file_data is None:
            print(f"No content found for file_id: {file_id}")
            continue  # Skip this iteration and proceed with the next file_id

        file_data_bytes = file_data.read()
        file_like_object = io.BytesIO(file_data_bytes)

        # Create a DataFrame from the file-like object and append
        df = pd.read_csv(file_like_object)
        loop_dfs.append(df)

    return loop_dfs

df_list = []
for messages, content, model, assistant in agent_responses:
    dataframes = create_dataframes_from_messages(messages, client)
    assistant_id = assistant.id
    df_list.append([dataframes, model, assistant_id])

# Capture the validation data scores
val_data_df_dict = {}
val_failures = []

# Loop through and capture validation data output
for item in df_list:
    try:
        df1 = pd.DataFrame(item[0][0]) 
        print(df1)
        if 'row_id' not in df1.columns:
            df1 = df1.reset_index().rename(columns={'index': 'row_id'})
        model = item[1]
        # Extract the first three letters of the model and the fold_id value
        key = model
        # Add the DataFrame to the dictionary with the generated key
        val_data_df_dict[key] = df1
        
    except:
        assistant_model = item[1]
        val_failures.append([assistant_model])
        
# Display failed data returns
print('assistants which failed to return a scored training data dataframe:')
print(val_failures)


# Capture the meta training data
test_data_df_dict = {}
test_failures = []

# Loop through and capture testing data output
for item in df_list:
    try:
        df1 = pd.DataFrame(item[0][1]) 
        print(df1)
        if 'row_id' not in df1.columns:
            df1 = df1.reset_index().rename(columns={'index': 'row_id'})
        model = item[1]
        # Extract the first three letters of the model and the fold_id value
        key = model
        # Add the DataFrame to the dictionary with the generated key
        test_data_df_dict[key] = df1
        
    except:
        assistant_model = item[1]
        test_failures.append([assistant_model])
        
# Display failed data returns
print('assistants which failed to return a scored training data dataframe:')
print(test_failures)


# create a target df to join everything to
list_of_val_keys = list(val_data_df_dict.keys())
first_val_key = list_of_val_keys[0]
meta_val_data = val_data_df_dict[first_val_key]

# Loop through the DataFrames in the dictionary, joining each to the label
for key in val_data_df_dict:
    if key != first_val_key and key not in val_failures:
        # get each dataframe
        cols_to_join = val_data_df_dict[key]
        # Join with the initial DataFrame on 'row_id'
        meta_val_data = meta_val_data.merge(cols_to_join, on='row_id', how='left')
        print(f'joined to {key}')

# add back label
val_label_df = encoded_train[['row_id', f'{target}']]
meta_val_data = meta_val_data.merge(val_label_df, on='row_id', how='left')

display(meta_val_data)   

# create a target df to join everything to
list_of_keys = list(test_data_df_dict.keys())
first_key = list_of_keys[0]
meta_training_data = test_data_df_dict[first_key]

# Loop through the DataFrames in the dictionary, joining each to the label
for key in test_data_df_dict:
    if key != first_key and key not in test_failures:
        # get each dataframe
        cols_to_join = test_data_df_dict[key]
        # Join with the initial DataFrame on 'row_id'
        meta_training_data = meta_training_data.merge(cols_to_join, on='row_id', how='left')
        print(f'joined to {key}')

# add back label
label_df = train_set[['row_id', f'{target}']]
meta_training_data = meta_training_data.merge(label_df, on='row_id', how='left')

display(meta_training_data)   

# save the meta training file
meta_training_data.to_csv('meta_train_df.csv', index=False)
meta_train_df = pd.read_csv('meta_train_df.csv')

# save the meta validation file (acting as inference data)
meta_val_data.to_csv('meta_val_df.csv', index=False)
meta_val_df = pd.read_csv('meta_val_df.csv')

import pandas as pd
from sklearn.metrics import accuracy_score

def calculate_model_accuracies(df):
    model_accuracy_dict = {}
    # Filter columns that contain probability predictions
    prediction_columns = [col for col in df.columns if "_prob" in col]
    
    for col in prediction_columns:
        # Assuming binary classification with 0.5 threshold
        predicted_classes = df[col].apply(lambda x: 1 if x >= 0.5 else 0)
        actual_classes = df[f'{target}']
        # Calculate accuracy
        accuracy = accuracy_score(actual_classes, predicted_classes)
        # Extract model name from column name 
        model_name = col.split("_prob")[0]
        model_accuracy_dict[model_name] = accuracy
    
    return model_accuracy_dict

accuracy_dict = calculate_model_accuracies(meta_val_df)
accuracy_dict

# Convert the dictionary into a DataFrame
base_model_accuracy_df = pd.DataFrame.from_dict(accuracy_dict, orient='index').reset_index()
base_model_accuracy_df.columns = ['Model', 'Accuracy_base']
base_model_accuracy_df.to_csv('base_model_accuracy.csv', index=False)

base_model_accuracy_df
