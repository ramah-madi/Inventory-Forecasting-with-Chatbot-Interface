
# Inventory Forecasting Project 📊

🚀 What is this app for?

This app is designed to help you make accurate predictions about future trends or behaviors using your time-series data.

🔍 What can I do with my data?

Once you upload your CSV file, you can visualize your time-series data, perform exploratory data analysis (EDA), build machine learning models for forecasting, and gain insights to make informed decisions.

Note: When the chatbot asks you for date and target column names make sure to provide the names as they written in the data due to case sensitivity.

## Setup
1. Clone the repository:
```python
git clone https://github.com/your-username/your-repository.git
```

2. Navigate to the project directory:
```python
cd your-repository
```

3. (Optinal) create new enviroment then activate it:
```python
python -m venv env
env\Scripts\activate
```

4. Install the required dependencies using pip:
```python
pip install -r requirements.txt
```

## Running the Application
### main.py
Run the server using the following command:
```python 
uvicorn server.api:app --reload
```
Note: api logic `api.py` in server folder.

### Chatbot Interface (chatbot.py)
run the chatbot interface using the following command:
```python
streamlit run chatbot.py
```

## Deployment
To deploy the app there is two steps:

1. Deploy the server using koyeb using this link https://app.koyeb.com:

     - login with github account.
     - choose web service.
     - select your repository project.
     - in configuration service select builder and toggle the override for run command and type: 
     ```python 
     uvicorn service.api:app --host 0.0.0.0
     ```
      change aother configurations based on your needs, then click on deploy.



2. Deploy the chatbot on streamlit:
   
    Note: before deploying the chatbot make sure to update the server links.

     - click on deploy in the right top corner (will appear when running the chatbot.py locally).
     - fill the requirement fileds.
     - click on Deploy! 







