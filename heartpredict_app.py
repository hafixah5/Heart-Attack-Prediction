# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 14:02:56 2022

@author: fizah
"""



import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import pickle
import os

#%% Constants

MODEL_PATH = os.path.join(os.getcwd(),'pickle files', 'best_estimator.pkl')

#%% Model deployment


with open(MODEL_PATH, 'rb') as file:
    model = pickle.load(file)
    

#%%

image = Image.open('header.jpg')
st.image(image)


st.title('Prediction of Heart Attack')
name = st.text_input("Name:")
age = st.number_input("Age:")
cp = st.selectbox("Chest pain type",('0','1','2','3'))
with st.expander("See Explanation"):
    st.write("""
              Value 0: Typical angina \n
Value 1: Atypical angina \n
Value 2:  Non-anginal pain \n
Value 3:  Asymptomatic\n
""")
trtbps = st.number_input("Resting blood pressure (mm Hg):")
chol = st.number_input("Cholesterol level (mg/dL):")
thalachh = st.number_input("Maximum heart rate:")
oldpeak = st.number_input("Oldpeak:") 
with st.expander("See Explanation"):
    st.write("Oldpeak is ST depression during exercise relative to resting")
thall = st.number_input("Thallium Heart Rate")
submit = st.button('Predict')
if submit:
    prediction = model.predict(np.expand_dims([age, cp, trtbps, chol, thalachh, oldpeak, thall],axis=0))
    if prediction == 0:
        st.write('Congratulations,',name, '!','You have a low risk of heart attack!')
        image = Image.open('output0.jpg')
        st.image(image)
      
    else:
        st.write('Warning, ',name, '!', 'You might contract heart attack.')
        with st.expander("More Info"):
            st.write("""
         Consult a doctor. \n
         Eat healthy foods. \n
         Keep active. \n
         Don't gain more weight than recommended.\n
         Avoid tobacco use \n
         """)
        image = Image.open('output1_tips.jpg')
        st.image(image)
        











