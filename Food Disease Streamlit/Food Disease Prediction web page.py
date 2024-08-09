# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 20:49:54 2024

@author: sindh
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle
import streamlit as st
import os





def diabetes_prediction(input_data):
    
  # Provide the correct absolute path to the 'Outbreaks.csv' file
  csv_file_path = r'E:\Food Disease Streamlit\Outbreaks.csv'  # Using raw string
  model_file_path = r'E:\Food Disease Streamlit\trained_model.sav'  # Using raw string

  # Verify the CSV file path
  print(f"Loading data from: {csv_file_path}")

  try:
      # Load the data
      data = pd.read_csv(csv_file_path)
      print("Data loaded successfully.")
  except FileNotFoundError:
      print(f"CSV file not found: {csv_file_path}")
      raise

  # Separate LabelEncoders for each categorical column
  encoders = {}
  for column in ['Month', 'State', 'Location', 'Food', 'Ingredient', 'Species', 'Serotype/Genotype', 'Status']:
      le = LabelEncoder()
      # Handle unknown values by ignoring them during transformation
      le.fit(data[column].astype(str))  # Fit the encoder on string values to handle potential mixed types
      data[column] = le.transform(data[column].astype(str))
      encoders[column] = le

  # Assuming the input data is given as a tuple
  input_data = (1998, 'January', 'California', 'Private Home/Residence', 'Lasagna, Unspecified', 'Eggs, Other', 'Salmonella enterica', 'Enteritidis', 'Confirmed', 3, 0)

  # Convert input_data to numpy array
  input_data_as_numpy_array = np.asarray(input_data, dtype=object)

  # Transform the categorical input data using the fitted encoders, handling unknown values
  for i, column in enumerate(['Month', 'State', 'Location', 'Food', 'Ingredient', 'Species', 'Serotype/Genotype', 'Status']):
      try:
          input_data_as_numpy_array[i+1] = encoders[column].transform([input_data_as_numpy_array[i+1]])[0]
      except ValueError:
          # Handle the case where the value is not seen during training
          input_data_as_numpy_array[i+1] = -1  # Or any other strategy to handle unknown values

  # Convert all elements to numeric values explicitly
  input_data_as_numpy_array = input_data_as_numpy_array.astype(float)

  # Reshape the array as we are predicting for one instance
  input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

  # Verify the model file path
  print(f"Loading model from: {model_file_path}")

  try:
      with open(model_file_path, 'rb') as file:
          loaded_model = pickle.load(file)
      print("Model loaded successfully.")
  except FileNotFoundError:
      print(f"Model file not found: {model_file_path}")
      raise

  # Make prediction
  prediction = loaded_model.predict(input_data_reshaped)
  print("Prediction:", prediction)

def main():
    st.title("Food Disease Prediction web page")
    
    year = st.text_input('Year')
    month = st.text_input('Month')
    state = st.text_input('State')
    location = st.text_input('Location')
    food = st.text_input('Food')
    ingredient = st.text_input('Ingredient')
    species = st.text_input('Species')
    serotype_genotype = st.text_input('Serotype/Genotype')
    status = st.text_input('Status')
    hospitalizations = st.text_input('Hospitalizations')
    fatalities = st.text_input('Fatalities')
    
    illness = ''
    
    if st.button("Illness Test Results"):
        illness = diabetes_prediction([year, month, state, location, food, ingredient, species, serotype_genotype, status, hospitalizations, fatalities])
    st.success(illness)

if __name__ == '__main__':
    main()
