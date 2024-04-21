**ODSC_East_4.25.24**
<br><br>
OpenAI’s API allows users to programmatically create custom GPTs, referred to as Assistants, which can be instructed to write and execute code on provided data. This opens many exciting possibilities in data science, in particular the use of multiple Assistants to help build large scale, powerful machine learning ensemble methods that might otherwise be unfeasible.
<br><br>
Model stacking is an advanced machine learning technique where multiple base models, typically of different types, are trained on the same data and their predictions used as input for a final ""meta-model"". While it is a powerful technique, stacking is generally impractical for most data scientists due to its heavy resource requirements and time-consuming architecture. However, by creating multiple AI Assistants through the API, these types of multi-model ensembles can be easily and quickly created.
<br><br>
In this presentation, I will show how a single user with a beginner level knowledge of python can create a “swarm” of AI Assistants that train a series of models for use in a model-stacking ensemble classifier that outperforms traditional ML models on the same data. We will go over each step from getting set up with the API to orchestrating an AI swarm, to collecting their output for the final Meta model predictions.
<br><br>
**Prerequisits for presentation:**

Code and data set: Github
Your favorite Python computation environment: 
<br>
Colab (https://colab.research.google.com/ ) 
<br>
Anaconda (https://www.anaconda.com/download )
<br><br>
OpenAI Assistant playground:
1.	Go to the OpenAI website: OpenAI
2.	Navigate to the "Products" section in the top menu.
3.	Select "ChatGPT" or directly go to the ChatGPT page.
4.	On the ChatGPT page, you should see options to either sign up or log in. If you don’t have an account, you can click on the sign-up option to create a new account.
By signing up, you’ll gain access to OpenAI's services including the Assistant Playground, which is the user interface that allows you to interact with Assistants https://platform.openai.com/playground 
       

