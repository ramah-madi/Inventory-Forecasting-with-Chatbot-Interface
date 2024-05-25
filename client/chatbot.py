import streamlit as st
import requests
import pandas as pd
from datetime import datetime

#Title
st.title('AI Assistant for future forecasting 🤖')

#Welcoming message
st.write("Hello, 👋 I am your AI Assistant and I am here to help you with your forecasting problem.")

# Explanation sidebar
with st.sidebar:
    st.subheader("Welcome to our Forecasting Adventure!")
    st.write("🚀 **What is this app for?**")
    st.write("This app is designed to help you make accurate predictions about future trends or behaviors using your time-series data.")

    st.write("⏰ **What is time-series data?**")
    st.write("Time-series data consists of observations recorded at regular time intervals. Examples include stock prices, weather patterns, and sensor data collected over time. Analyzing time-series data allows us to identify patterns, trends, and seasonal variations, enabling us to make predictions about future values.")

    st.write("📊 **Statistical Models Forecasting:**")
    st.write("This option leverages statistical techniques such as Auto ARIMA (Auto-Regressive Integrated Moving Average) and Holt-Winters Exponential Smoothing. Auto ARIMA is particularly beneficial for capturing linear relationships and seasonality within time-series data. Unlike traditional ARIMA, Auto ARIMA automatically determines the optimal parameters for the model, making it more efficient and user-friendly. On the other hand, Holt-Winters Exponential Smoothing is well-suited for handling seasonal and trend data, providing valuable insights into forecasting future inventory levels.")
    
    st.write("🤖 **Machine Learning Models Forecasting:**")
    st.write("Machine learning models, including Gradient Boosting and MLP Regressor, are utilized for forecasting purposes. Gradient Boosting is an ensemble learning technique that sequentially builds multiple decision trees, each one correcting the errors of its predecessor, thus effectively capturing complex relationships in the data. This method is highly effective in improving predictive accuracy and handling various types of data. MLP Regressor, also known as Multi-Layer Perceptron Regressor, represents a type of artificial neural network adept at recognizing and learning non-linear patterns within data, providing valuable insights into forecasting inventory levels.")
    
    st.write("📊 **Why start with a CSV file?**")
    st.write("A CSV file is a convenient format for storing time-series data, making it easy to upload and analyze. By uploading your CSV file, you'll be ready to embark on your forecasting journey.")
    
    st.write("🔍 **What can I do with my time-series data?**")
    st.write("Once you upload your CSV file, you can visualize your time-series data, perform exploratory data analysis (EDA), build machine learning models for forecasting, and gain insights to make informed decisions.")
    
    st.write("🤖 **How does it work?**")
    st.write("1. **Upload your CSV file:** Click on the 'Upload CSV file' button and select your time-series dataset.")
    st.write("2. **Specify the date and target columns:** Identify the date column and the target variable you want to forecast.")
    st.write("3. **Build forecasting models:** Use machine learning algorithms to train forecasting models and predict future values based on historical data.")
    st.write("4. **Evaluate model performance:** Analyze forecast accuracy metrics and visualize future predictions to make data-driven decisions.")
    
    st.write("📝 **How can I get started?**")
    st.write("1. Upload your CSV file.")
    st.write("2. Specify the date and target columns.")
    st.write("3. Build forecasting models.")
    st.write("4. Evaluate model performance and explore future predictions.")
    
    st.divider()
    
    st.caption("<p style ='text-align:center'>Made by Ramah Madi</p>", unsafe_allow_html=True)


# Function to interact with the ML forecast API
def ml_forecast(data_file, start, end, date_column, target_column):
    files = {'file': data_file}
    data = {
        'start': start,
        'end': end,
        'date_column': date_column,
        'target_column': target_column
    }
    response = requests.post('https://server-api-chatbot-hopy-1b452263.koyeb.app/api/mlForecast/', files=files, data=data)
    return response.json()

# Function to interact with the Stats forecast API
def stats_forecast(data_file, start, end, date_column, target_column):
    files = {'file': data_file}
    data = {
        'start': start,
        'end': end,
        'date_column': date_column,
        'target_column': target_column
    }
    response = requests.post('https://server-api-chatbot-hopy-1b452263.koyeb.app/api/statsForecast/', files=files, data=data)

    return response.json()

def type_of_forecast(name, forecast_function):
    st.write(f"🤖 You've chosen {name} Forecast.")
    st.write("----")
    st.write("🤖 First, please upload your data file:")
    data_file = st.file_uploader("Upload CSV file", type=['csv'])

    if data_file is not None:
        # Load the CSV file into a DataFrame
        data = pd.read_csv(data_file)

        st.write("🤖 Great! Now, I need some details about your data.")
        
        start = st.text_input("Please enter the start date of the forecast period (MM-DD-YYYY e.g.01-01-2024):", help="Please enter the start date for the forecast period.")
        end = st.text_input("Please enter the end date of the forecast period (MM-DD-YYYY e.g02-01-2024):", help="Please enter the end date for the forecast period.")

        if start and end:
            # Validate date input
            try:
                start_date = datetime.strptime(start, '%m-%d-%Y')
                end_date = datetime.strptime(end, '%m-%d-%Y')
            except ValueError:
                st.error("🤖 Error: Please enter valid dates in MM-DD-YYYY format.")
                return

            st.write("🤖 Thanks! Now, could you tell me the name of the date column in your dataset?")
            date_column = st.text_input("Please enter the name of the date column:", help="This is the column that contains the date information.")
            
            if date_column:
                if date_column not in data.columns:
                    st.error(f"🤖 Error: The column '{date_column}' does not exist in the dataset.")
                    return

                st.write("🤖 Got it! Finally, I need to know the name of the target column you want to forecast.")
                target_column = st.text_input("Please enter the name of the target column:", help="This is the column you want to make predictions for.")
                
                if target_column:
                    if target_column not in data.columns:
                        st.error(f"🤖 Error: The column '{target_column}' does not exist in the dataset.")
                        return
                    
                    if st.button("Confirm"):
                        st.write(f"🤖 Performing {name} Forecast...")
                        try:
                            # Reopen the file to ensure it is sent correctly
                            data_file.seek(0)
                            response = forecast_function(data_file, start_date, end_date, date_column, target_column)
                        except Exception as e:
                            st.error(f"🤖 Error: Forecast function failed. Details: {e}")
                            return
                        
                        # Check if response contains 'Test_Accuracy' key
                        if 'Test_Accuracy' in response:
                            st.write("🤖 Forecasting results:")
                            
                            # Display accuracy metrics in a table
                            st.subheader("Accuracy Metrics:")
                            accuracy_df = pd.DataFrame(response['Test_Accuracy']).T
                            st.dataframe(accuracy_df, width=600)  # Increase the width of the DataFrame

                            # Description of accuracy metrics
                            st.write("MAE measures the average size of the errors in a set of predictions. It tells you how close, on average, the predictions are to the actual values. Lower MAE values indicate better accuracy. For example, if the actual temperatures for a day are [75, 80, 85, 90] degrees Fahrenheit, and the predicted temperatures are [72, 79, 86, 88], the MAE would be (|75-72| + |80-79| + |85-86| + |90-88|) / 4 = 3.5 degrees Fahrenheit.")
                            st.write("MSE measures the average of the squares of the errors, giving more weight to larger errors. It provides a detailed view of how far the predictions are from the actual values. Lower MSE values also indicate better accuracy. Using the same example of temperatures, the MSE would be ((75-72)^2 + (80-79)^2 + (85-86)^2 + (90-88)^2) / 4 = 5.5 square degrees Fahrenheit.")
                            
                            # Display future forecast plots
                            st.subheader("Future Forecast:")
                            for model, forecast in response['Future_Forecast'].items():
                                st.write(f"- {model}:")
                                st.line_chart(forecast, use_container_width=True)  # Plot the forecast using Streamlit's line_chart
                                st.write("X-axis represents Days, Y-axis represents Demand ( the quantity of items will be in demand by the customers)")
                        else:
                            st.error("🤖 Error: Failed to retrieve forecasting results. Please try again.")


# Main function to run the chatbot interface
def main():
    
    # Ask the user to choose between statistical or machine learning forecast
    forecast_type = st.radio("Choose the type of forecast:", ("Statistical Models Forecast", "Machine Learning Models Forecast"), index=0)

    # Only display file upload prompt if a forecast type has been selected
    if forecast_type:
        # Call the appropriate function based on user choice
        if forecast_type == "Machine Learning Models Forecast":
            type_of_forecast("Machine Learning", ml_forecast)
        elif forecast_type == "Statistical Models Forecast":
            type_of_forecast("Statistical", stats_forecast)

if __name__ == "__main__":
    main()
    