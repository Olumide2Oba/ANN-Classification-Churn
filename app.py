import streamlit as st          
import numpy as np     
import pandas as pd       
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder  
import tensorflow as tf    
##from tensorflow.keras.models import load_model, save_model       
import pickle    
import os        



### Load the trained model, Scaler, OneHotEncoder... 
model = tf.keras.models.load_model('model.h5')  # Fixed: Use `tf.keras.models.load_model`
os.makedirs('saved_model', exist_ok=True)
model.save('saved_model/my_model.keras')      # Save in SavedModel format

### Load the encoder, scaler, OneHotEncoder 
with open('onehot_encoder_geography.pkl', 'rb') as file:
    onehot_encoder_geography = pickle.load(file)
    
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)  # Fixed: Variable name consistency
    
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file) 

## Streamlit App 
st.title('Customer Churn Prediction')

# Input widgets
Geography = st.selectbox('Geography', onehot_encoder_geography.categories_[0])
Gender = st.selectbox('Gender', label_encoder_gender.classes_)
Age = st.slider('Age', 18, 92)
Balance = st.number_input('Balance')
Credit_Score = st.number_input('Credit Score')
Estimated_Salary = st.number_input('Estimated Salary')  # Fixed: Variable name consistency
Tenure = st.slider('Tenure', 0, 10)
Num_of_Product = st.slider('Number of Products', 1, 4)
Has_Credit_Card = st.selectbox('Has Credit Card', [0, 1])
Is_Active_Member = st.selectbox('Is Active Member', [0, 1])

## Prepare the input data 
input_data = pd.DataFrame({
    'CreditScore': [Credit_Score],
    'Gender': [label_encoder_gender.transform([Gender])[0]],
    'Age': [Age],
    'Tenure': [Tenure],
    'Balance': [Balance],
    'NumOfProducts': [Num_of_Product],  # Fixed: Column name consistency (check model training)
    'HasCrCard': [Has_Credit_Card],     # Fixed: Column name consistency
    'IsActiveMember': [Is_Active_Member],
    'EstimatedSalary': [Estimated_Salary]
})

# Example input
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Assume this is the category you want to encode
geo_input = pd.DataFrame([['France']], columns=['Geography'])

# Fit or load your encoder
onehot_encoder_geography = OneHotEncoder()
onehot_encoder_geography.fit([['France'], ['Germany'], ['Spain']])  # example categories

# Transform input
geo_encoded = onehot_encoder_geography.transform(geo_input).toarray()

# Now this should work
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geography.get_feature_names_out(['Geography']))

## Combine one-hot encoded columns with input data 
input_data = pd.concat([input_data, geo_encoded_df], axis=1)

## Scale the data 
input_data_scaled = scaler.transform(input_data)

## Predict Churn
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

st.write(f'Churn Probability: {prediction_proba:.2f}')

if prediction_proba > 0.5:
    st.error('The Customer is likely to Churn')
else:
    st.success('The Customer is not likely to Churn')