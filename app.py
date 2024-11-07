import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pandas as pd
import pickle

#loading the model
model = tf.keras.models.load_model('model.h5')

#load the encoder
with open('onehot_encoder_geo.pkl','rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender = pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file)


#streamlit app
st.title('Customer Churn Prediction')

#streamlit app buttons for input
geography = st.selectbox('Geography',onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender',label_encoder_gender.classes_)
age = st.slider('Age',18,92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure  = st.slider('Tenure',0,10)
num_of_products = st.slider('Number of Products',1,4)
has_cr_card = st.selectbox('Has Credit Card',[0,1])
is_active_member = st.selectbox('Is Active Member',[0,1])

#creating dictionary of inputs
input_data = pd.DataFrame({
    'CreditScore':[credit_score],
    'Gender':[label_encoder_gender.transform([gender])[0]],
    'Age':[age],
    'Tenure': [tenure],
    'Balance':[balance],
    'NumOfProducts':[num_of_products],
    'HasCrCard':[has_cr_card],
    'IsActiveMember':[is_active_member],
    'EstimatedSalary':[estimated_salary]
})

#encoding geography
geo_encode = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encode,columns = onehot_encoder_geo.get_feature_names_out(['Geography']))

#creating final_df
input_df = pd.concat([input_data.reset_index(drop =True),geo_encoded_df],axis = 1)

#scaling the input df
input_df_scaled = scaler.transform(input_df)

#prediction
prediction = model.predict(input_df_scaled)
predict_proba = prediction[0][0]

if predict_proba>0.5:
    st.write(f'The customer is likely to churn.Churn Probablity is {round(predict_proba*100,2)}%')
else:
    st.write('The customer is likely to stay')