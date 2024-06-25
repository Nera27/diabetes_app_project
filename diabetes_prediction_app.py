import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Function to load the dataset
#@st.cache_data
def load_data():
    diabetes_dataset = pd.read_csv('/Users/veneraheddergott/desktop/diabetes.csv')
    return diabetes_dataset

# Load the data
diabetes_dataset = load_data()

# Sidebar for navigation
st.sidebar.title("Navigation")
options = st.sidebar.radio("Go to", ["Dataset Information", "Exploratory Data Analysis", "Model Training and Evaluation", "Prediction"])

# Title and description
st.title('Diabetes Prediction App')
st.write('This app uses a logistic regression model to predict whether a patient has diabetes based on their health parameters.')

# Data preparation and model training
X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, stratify=Y, random_state=2)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)

if options == "Dataset Information":
    st.subheader('Dataset Information')
    buffer = st.empty()
    buffer.text(str(diabetes_dataset.info()))

    st.subheader('First 10 Rows of the Dataset')
    st.write(diabetes_dataset.head(10))

    st.subheader('Missing Values in the Dataset')
    st.write(diabetes_dataset.isnull().sum())

elif options == "Exploratory Data Analysis":
    st.subheader('Exploratory Data Analysis')

    st.write(diabetes_dataset.describe())

    st.write('### Histograms of All Features')
    fig, ax = plt.subplots(figsize=(15, 10))
    diabetes_dataset.hist(bins=15, ax=ax)
    st.pyplot(fig)

    st.write('### Boxplot of All Features')
    fig, ax = plt.subplots(figsize=(15, 10))
    sns.boxplot(data=diabetes_dataset, orient='h', ax=ax)
    st.pyplot(fig)

    st.write('### Pairplot of All Features')
    fig = sns.pairplot(diabetes_dataset, hue='Outcome')
    st.pyplot(fig)

    st.write('### Distribution of Diabetes Outcome')
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x='Outcome', data=diabetes_dataset, ax=ax)
    st.pyplot(fig)

    outcome_means = diabetes_dataset.groupby('Outcome').mean()
    st.write('### Mean Values of Features Grouped by Outcome')
    st.write(outcome_means)

    fig, ax = plt.subplots(figsize=(12, 6))
    outcome_means.T.plot(kind='bar', ax=ax)
    st.pyplot(fig)

    st.write('### Correlation Matrix of All Features')
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(diabetes_dataset.corr(), annot=True, cmap='coolwarm', linewidths=0.5, ax=ax)
    st.pyplot(fig)

elif options == "Model Training and Evaluation":
    st.subheader('Model Preparation and Training')

    st.subheader('Model Evaluation')

    X_train_prediction = model.predict(X_train)
    training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
    st.write('Training Data Accuracy with Logistic Regression:', training_data_accuracy)

    X_test_prediction = model.predict(X_test)
    test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
    st.write('Test Data Accuracy with Logistic Regression:', test_data_accuracy)

    conf_matrix = confusion_matrix(Y_test, X_test_prediction)
    st.write('### Confusion Matrix')
    st.write(conf_matrix)

    st.write('### Classification Report')
    class_report = classification_report(Y_test, X_test_prediction)
    st.write(class_report)

    st.write('### Confusion Matrix Heatmap')
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    st.pyplot(fig)

elif options == "Prediction":
    st.subheader('Diabetes Prediction')
    st.write('Enter the parameters to get a prediction.')

    pregnancies = st.slider('Number of Pregnancies', 0, 17, 1)
    glucose = st.slider('Glucose', 0, 200, 100)
    blood_pressure = st.slider('Blood Pressure', 0, 122, 70)
    skin_thickness = st.slider('Skin Thickness', 0, 99, 20)
    insulin = st.slider('Insulin', 0, 846, 79)
    bmi = st.slider('BMI', 0.0, 67.1, 25.0)
    diabetes_pedigree_function = st.slider('Diabetes Pedigree Function', 0.0, 2.42, 0.5)
    age = st.slider('Age', 21, 81, 33)

    user_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]])
    user_data = scaler.transform(user_data)

    st.write('### Input Parameters')
    st.write(pd.DataFrame(user_data, columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']))

    prediction = model.predict(user_data)
    prediction_prob = model.predict_proba(user_data)

    st.write('### Prediction Result')
    if prediction[0] == 1:
        st.write('The model predicts that the patient has diabetes.')
    else:
        st.write('The model predicts that the patient does not have diabetes.')

    st.write('### Prediction Probability')
    st.write(f'Probability of having diabetes: {prediction_prob[0][1]:.2f}')
    st.write(f'Probability of not having diabetes: {prediction_prob[0][0]:.2f}')
#