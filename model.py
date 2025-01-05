import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

df = pd.read_csv("model_dataset.csv")
df1 = df[[
    #'Marital status', 'Application mode', 'Application order', 'Course',
       #'Daytime/evening attendance', 'Previous qualification',
    #'Nacionality',
       #'Mother's qualification', 'Father's qualification',
       #'Mother's occupation', 'Father's occupation',
    'Displaced',
       #'Educational special needs',
    #'Debtor',
    #'Tuition fees up to date',
       'Gender', 'Scholarship holder', 'Age at enrollment',
    #'International',
       'Curricular units 1st sem (credited)',
       'Curricular units 1st sem (enrolled)',
       'Curricular units 1st sem (evaluations)',
       'Curricular units 1st sem (approved)',
       'Curricular units 1st sem (grade)',
       'Curricular units 1st sem (without evaluations)',
       'Curricular units 2nd sem (credited)',
       'Curricular units 2nd sem (enrolled)',
       'Curricular units 2nd sem (evaluations)',
       'Curricular units 2nd sem (approved)',
       'Curricular units 2nd sem (grade)',
       'Curricular units 2nd sem (without evaluations)',
    #'Unemployment rate',
       #'Inflation rate', 'GDP',
    'Target']]

# Create new features for 1st sem
df1['Performance Rate 1st sem'] = df1['Curricular units 1st sem (approved)'] / df1['Curricular units 1st sem (enrolled)'] #Performance
df1['Engagement Rate 1st sem'] = df1['Curricular units 1st sem (evaluations)'] / df1['Curricular units 1st sem (enrolled)'] #Engagement
df1['Credit Completion Rate 1st sem'] = df1['Curricular units 1st sem (credited)'] / df1['Curricular units 1st sem (enrolled)']

# Flag students at risk
df1['Low Performance Rate 1st sem'] = (df1['Performance Rate 1st sem'] < 0.5).astype(int)
df1['Low Engagement Rate 1st sem'] = (df1['Engagement Rate 1st sem'] < 0.7).astype(int)
#dropping the student with 0 curricular unit in 1st sem
df1.drop(df1[df1['Curricular units 1st sem (enrolled)'] == 0].index, inplace=True)

# Create new features for 2nd sem
df1['Performance Rate 2nd Sem'] = df1['Curricular units 2nd sem (approved)'] / df1['Curricular units 2nd sem (enrolled)'] #Performance
df1['Engagement Rate 2nd Sem'] = df1['Curricular units 2nd sem (evaluations)'] / df1['Curricular units 2nd sem (enrolled)'] #Engagement
df1['Credit Completion Rate 2nd Sem'] = df1['Curricular units 2nd sem (credited)'] / df1['Curricular units 2nd sem (enrolled)']

# Flag students at risk
df1['Low Performance Rate 2nd Sem'] = (df1['Performance Rate 2nd Sem'] < 0.5).astype(int)
df1['Low Engagement Rate 2nd Sem'] = (df1['Engagement Rate 2nd Sem'] < 0.7).astype(int)

# Normalize numerical features


numerical_cols = ['Age at enrollment', 'Curricular units 1st sem (credited)',
       'Curricular units 1st sem (enrolled)',
       'Curricular units 1st sem (evaluations)',
       'Curricular units 1st sem (approved)',
       'Curricular units 1st sem (grade)',
       'Curricular units 1st sem (without evaluations)',
       'Curricular units 2nd sem (credited)',
       'Curricular units 2nd sem (enrolled)',
       'Curricular units 2nd sem (evaluations)',
       'Curricular units 2nd sem (approved)',
       'Curricular units 2nd sem (grade)',
       'Curricular units 2nd sem (without evaluations)',
       'Performance Rate 1st sem', 'Engagement Rate 1st sem',
       'Credit Completion Rate 1st sem', 'Performance Rate 2nd Sem',
       'Engagement Rate 2nd Sem', 'Credit Completion Rate 2nd Sem']

scaler = MinMaxScaler()
df1[numerical_cols] = scaler.fit_transform(df1[numerical_cols])

df1_DropOut_Graduate = df1[df1['Target'].isin(['Dropout', 'Graduate'])]

df1_DropOut_Graduate['Target'] = df1_DropOut_Graduate['Target'].map({'Dropout': 1, 'Graduate': 0})

x = df1_DropOut_Graduate.drop('Target', axis=1)
y = df1_DropOut_Graduate['Target']

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=.2, random_state=42, stratify=y)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

# Predict using Random Forest
y_pred = model.predict(x_test)
y_prob = model.predict_proba(x_test)[:, 1]

#evaluate
model_accuracy = accuracy_score(y_test,y_pred)

joblib.dump(model, 'dropout_model.pkl')