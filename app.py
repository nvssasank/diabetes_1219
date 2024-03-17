import numpy as np
import pandas as pd
import plotly.express as px
import json

import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from PIL import Image
import streamlit as st


KNN=pickle.load(open("KNN.sav","rb"))
LR=pickle.load(open("LR.sav","rb"))
RF=pickle.load(open("RF.sav","rb"))

#set tab name and favicon
st.set_page_config(page_title="Diabetes Detection", page_icon="💉", layout='centered', initial_sidebar_state='auto')



#Create Title
st.write("""
# Diabetes Detection 
Predict if someone has diabetes or not using Machine Learning
""")
def loadData():
    df=pd.read_csv("diabetes.csv")
    #clean missing values with reference to their distribution
    df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']]=df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)
    df['Glucose'].fillna(df['Glucose'].mean(), inplace = True)
    df['BloodPressure'].fillna(df['BloodPressure'].mean(), inplace = True)
    df['SkinThickness'].fillna(df['SkinThickness'].median(), inplace = True)
    df['Insulin'].fillna(df['Insulin'].median(), inplace = True)
    df['BMI'].fillna(df['BMI'].median(), inplace = True)
    return df

df=loadData()


#Set subheader and display user input
st.header('User Input and Prediction:')
st.markdown('### Input:')

 #sidebar 
st.sidebar.header('User Input')
st.sidebar.write('Predict whether you have diabetes or not by entering the parameters. The results are located at the bottom of the page')
option = st.sidebar.selectbox('Select your Machine Learning Model', ('K Nearest Neighbors', 'Logistic Regression', 'Random Forest'))






#Get User input
def getUserInfo():
    pregnancies = st.sidebar.text_input('Pregnancies had', '3')
    glucose = st.sidebar.text_input('Plasma Glucose Concentration (mg/dl)', '117')
    bloodPressure = st.sidebar.text_input('Diastolic Blood Pressure (mm Hg)', '72')
    skinThickness = st.sidebar.text_input('Triceps skin fold thickness (mm)', '23')
    insulin = st.sidebar.text_input('Serum Insulin (U/ml)', '30.0')
    bmi = st.sidebar.text_input('Body Mass Index (BMI)', '32.0')
    diabetesPedigreeFunction = st.sidebar.text_input('Diabetes Pedigree Function', '0.3725')
    age = st.sidebar.text_input('Age', '29')
    
    # Convert inputs to appropriate data types
    pregnancies = int(pregnancies)
    glucose = float(glucose)
    bloodPressure = float(bloodPressure)
    skinThickness = float(skinThickness)
    insulin = float(insulin)
    bmi = float(bmi)
    diabetesPedigreeFunction = float(diabetesPedigreeFunction)
    age = int(age)
    
    # Store into dictionary with lowercase keys
    userData = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': bloodPressure,
        'SkinThickness': skinThickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': diabetesPedigreeFunction,
        'Age': age
    }

    # Transform to DataFrame
    features = pd.DataFrame(userData, index=[0])
    jsonObject = json.dumps(userData, indent=4)
    st.json(jsonObject)

    return features

#Store user input to variable
userInput=getUserInfo()

#selection of ml models 
if option=='K Nearest Neighbors':
    accuracy='71%'
    prediction=KNN.predict(userInput)
    st.text('K Nearest Neighbours model Prediction: ')
    if prediction==1:
        st.warning("You have a probability of having diabetes. Please consult with your doctor")
    elif prediction==0:
        st.success("Congratulations! You Don't have Traces of diabetes")
elif option=='Logistic Regression':
    accuracy='73%'
    prediction=LR.predict(userInput)
    st.text('Logistic Regression model Prediction: ')
    if prediction==1:
        st.warning("You have a probability of having diabetes. Please consult with your doctor")
    elif prediction==0:
        st.success("Congratulations! You Don't have Traces of diabetes")
elif option=='Random Forest':
    accuracy='74%'
    prediction=RF.predict(userInput)
    st.text('Random Forest model Prediction: ')
    if prediction==1:
        st.warning("You have a probability of having diabetes. Please consult with your doctor")
    elif prediction==0:
        st.success("Congratulations! You Don't have Traces of diabetes")


KNN1 = KNN.predict(userInput)
LR1 = LR.predict(userInput)
RF1 = RF.predict(userInput)

st.markdown("""<style>div.row-widget.col-6{flex: 0 0 50%;max-width: 50%;}.fullScreenFrame{width: 100%;height: 100%;}</style>""", unsafe_allow_html=True)
st.markdown("""<style>div.row-widget.stRadio > div{flex: 0 0 50%;max-width: 50%;}</style>""", unsafe_allow_html=True)
predictionList=[KNN1[0],LR1[0],RF1[0]]

from collections import Counter
predictionList = [item for item, count in Counter(predictionList).items() if count > 1]

st.markdown("""<style>div.row-widget.stRadio > div{flex: 0 0 50%;max-width: 50%;}</style>""", unsafe_allow_html=True)
st.markdown("Over All Predicted Diagnosis: ", unsafe_allow_html=True)
for i in range(len(predictionList)):
    if predictionList[i] == 1:
        st.warning("You have a probability of having diabetes. Please consult with your doctor")      
    else:
        st.success("Congratulations! You Don't have Traces of diabetes")
