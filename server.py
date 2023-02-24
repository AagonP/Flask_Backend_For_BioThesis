from flask import Flask, request
from flask_cors import CORS, cross_origin

# Import libs to run model
import pandas as pd
import numpy as np

import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn import svm
from sklearn.linear_model import LogisticRegression
import pickle


df = pd.read_csv("PPG-BP dataset.csv",delimiter=",")
new_df = df.drop(['Num.','subject_ID', 'Diabetes' , 'cerebral infarction', 'cerebral infarction','cerebrovascular disease','BMI(kg/m^2)'], axis=1)
diseases = pd.DataFrame(df["Hypertension"].copy(), columns=['Hypertension'])
hypertension_encoder = OrdinalEncoder()
hyptertension_ordi = hypertension_encoder.fit_transform(diseases)
encoded_diseases = pd.DataFrame(hyptertension_ordi, columns=diseases.columns,index=diseases.index)
class_names = ['Normal', 'Prehypertension', 'Stage 1 hypertension','Stage 2 hypertension']
reshaped_encoded_diseases = encoded_diseases['Hypertension'].copy()
# Remove extra columns
X_df = new_df.drop("Hypertension",axis=1)
# Change Sex to 0,1
X_df['Sex(M/F)'].replace(['Female','Male'],[0,1],inplace=True)
X_train, X_test, y_train, y_test = train_test_split(X_df,reshaped_encoded_diseases,test_size=0.2, random_state=42,stratify=reshaped_encoded_diseases)
# Transform pipeline
# Scaling features
# Clean data

num_pipeline = Pipeline([
    ('std_scaler',StandardScaler())
])

num_attribs = list(X_df)

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
])
X_train_prepared = full_pipeline.fit_transform(X_train)



model = pickle.load(open("./Models/model.sav", 'rb'))
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route("/health")
@cross_origin()
def hello_world():
    return "<p>I'm working lol.</p>"

@app.route("/")
@cross_origin()
def root():
    return "<p>Flask Backend For BioThesis</p>"

@app.route('/api/predict', methods=['POST'])
@cross_origin()
def predict():
    class_names = ['Normal', 'Prehypertension', 'Stage 1 hypertension','Stage 2 hypertension']
    print(request.json)
    patient_data = request.json
    patient_data_list = list(patient_data.values())
    df_patient = pd.DataFrame([patient_data_list],columns=['Sex(M/F)','Age(year)','Height(cm)','Weight(kg)','Systolic Blood Pressure(mmHg)','Diastolic Blood Pressure(mmHg)','Heart Rate(b/m)'])
    
    test_data_prepared = full_pipeline.transform(df_patient)
    print(test_data_prepared)
    result = model.predict(test_data_prepared)
    print(result)
    
    return {'disease': class_names[int(result)]}
