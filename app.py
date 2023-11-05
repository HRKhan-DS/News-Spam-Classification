import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load the Random Forest model outside of any function
with open("RandomForestModel.pkl", "rb") as model_file:
    rf_model = pickle.load(model_file)

# Try reading the CSV file with different encodings
encodings_to_try = ['utf-8', 'latin-1', 'ISO-8859-1']

for encoding in encodings_to_try:
    try:
        df = pd.read_csv('clear_data.csv', encoding=encoding)
        break  # Break the loop if successful
    except UnicodeDecodeError:
        continue  # Try the next encoding

def main():
    global rf_model  # Declare rf_model as a global variable
    st.title("News Spam Classification")

    # Create an input text area for the user to enter news content
    input_data = st.text_area("Enter the news text:")

    if st.button("Classify"):
        # Make sure the input data is not empty
        if input_data:
            # Perform any preprocessing on input_data if necessary (e.g., vectorization)
            # Here, you should apply the same preprocessing steps used when training the model

            # Predict the class of the news
            prediction = rf_model.predict([input_data])

            # Display the prediction result
            if prediction[0] == 0:
                st.write('The News is classified as Real')
            else:
                st.write('The News is classified as Fake')
        else:
            st.write("Please enter some news text before classifying.")

    # Larger gap using multiple <br> tags
    st.markdown("<br><br><br>", unsafe_allow_html=True)

    # Instructions for Fake News Classifier
    st.write("News Spam Classification!")
    st.write("To determine whether the news is true or false, enter the name of the news author and the title in the input box.")
    st.write("Click the 'Classify' button, and the result (real or fake) will be displayed above.")

if __name__ == '__main__':
    main()

