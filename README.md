## Recipe Application with RAG ##
In this project, a recipe application has been prepared with the RAG method using Google Gemini 2.0-Flash. 
If the question asked by the user is in the source pdf it retrieves from source, else it finds the answer to the question from the internet in an agentic way.

## Technologies Used ##
*haystack

*google-ai-haystack

*streamlit

*dotenv

## Execution the Code ##
An .env file is created after pulling the code. File content should be like this:
GOOGLE_API_KEY = 'YOUR_API_KEY'
Then, the command
```
pip install -r requirements.txt
```
is written and the necessary libraries for the code to run are downloaded. Finally,
```
streamlit run recipe_app.py
```
is written to run the code.
