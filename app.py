import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import joblib

# Load pre-trained model
#model = joblib.load("dropout_model.pkl")

model = joblib.load("dropout_model.pkl")

# Streamlit App Title
st.title("Student Dropout Prediction App")
st.write("""
Upload a CSV file containing student data, and this app will predict whether each student is at risk of dropping out.
""")
features = ['Displaced',
            'Gender',
            'Scholarship holder',
            'Age at enrollment',
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
            'Performance Rate 1st sem',
            'Engagement Rate 1st sem',
            'Credit Completion Rate 1st sem',
            'Low Performance Rate 1st sem',
            'Low Engagement Rate 1st sem',
            'Performance Rate 2nd Sem',
            'Engagement Rate 2nd Sem',
            'Credit Completion Rate 2nd Sem',
            'Low Performance Rate 2nd Sem',
            'Low Engagement Rate 2nd Sem']



uploaded_file = st.file_uploader("Upload your CSV file here", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    data.drop("Target", axis=1)

    # Create new features for 1st sem
    data['Performance Rate 1st sem'] = data['Curricular units 1st sem (approved)'] / data['Curricular units 1st sem (enrolled)'] #Performance
    data['Engagement Rate 1st sem'] = data['Curricular units 1st sem (evaluations)'] / data['Curricular units 1st sem (enrolled)'] #Engagement
    data['Credit Completion Rate 1st sem'] = data['Curricular units 1st sem (credited)'] / data['Curricular units 1st sem (enrolled)']

    # Flag students at risk
    data['Low Performance Rate 1st sem'] = (data['Performance Rate 1st sem'] < 0.5).astype(int)
    data['Low Engagement Rate 1st sem'] = (data['Engagement Rate 1st sem'] < 0.7).astype(int)
    #dropping the student with 0 curricular unit in 1st sem
    data.drop(data[data['Curricular units 1st sem (enrolled)'] == 0].index, inplace=True)

    # Create new features for 2nd sem
    data['Performance Rate 2nd Sem'] = data['Curricular units 2nd sem (approved)'] / data['Curricular units 2nd sem (enrolled)'] #Performance
    data['Engagement Rate 2nd Sem'] = data['Curricular units 2nd sem (evaluations)'] / data['Curricular units 2nd sem (enrolled)'] #Engagement
    data['Credit Completion Rate 2nd Sem'] = data['Curricular units 2nd sem (credited)'] / data['Curricular units 2nd sem (enrolled)']

    # Flag students at risk
    data['Low Performance Rate 2nd Sem'] = (data['Performance Rate 2nd Sem'] < 0.5).astype(int)
    data['Low Engagement Rate 2nd Sem'] = (data['Engagement Rate 2nd Sem'] < 0.7).astype(int)

    st.write("### Dataset Summary")
    st.write(data.describe())
    
    #Filter Data 
    gender_filter=st.selectbox("Select Gender", options=["All", 1,0])
    scholarship_filter = st.selectbox("Select Scholarship Holder", options=["All", 1, 0])
    displaced_filter = st.selectbox("Select Displacement Status", options=["All", 1, 0])
    filtered_data = data.copy()
    if gender_filter != "All":
        filtered_data = data[data["Gender"] == gender_filter]
    if scholarship_filter != "All":
        filtered_data = data[data['Scholarship holder'] == scholarship_filter]
    if displaced_filter != "All":
        filtered_data = data[data['Displaced'] == displaced_filter]

    #Data Visualization
    col1, col2 = st.columns(2)

    with col1:
        st.write('Boxplot of Performance Rate for the 1st Semester')
        fig, ax = plt.subplots()
        sns.boxplot(x=filtered_data['Performance Rate 1st sem'], ax=ax)
        st.pyplot(fig)

    with col2:
        st.write('Boxplot of Performance Rate for the 2nd Semester')
        fig, ax = plt.subplots()
        sns.boxplot(x=filtered_data['Performance Rate 2nd Sem'], ax=ax)
        st.pyplot(fig)

    st.write("Age distribution")
    fig, ax = plt.subplots()
    sns.histplot(filtered_data["Age at enrollment"],kde=True, color="green", ax=ax)
    ax.set_title("Distribution of Age")
    ax.set_xlabel("Age")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)



    #Data Prediction
    if st.button("Predict"):
        try:

            predictions = model.predict(data[features])
            probabilities = model.predict_proba(data[features])[:,1]

            data["Dropout_Prediction"] = ["Yes" if pred == 1 else "No" for pred in predictions]
            data["Risk_Score"] = probabilities.round(2)

            st.write("### Review of the Predicted data:")
            st.dataframe(data[["Student ID", "Dropout_Prediction", "Risk_Score"]].head())
            data_dropout = data[data["Target"] == "Yes"]

            csv = data_dropout.to_csv(index=False)
        except Exception as e:
            st.error(f"An error occured during prediction: {e}")

        st.download_button(
            label = "Download Predictions as CSV",
            data = csv,
            file_name = "dropout_predictions.csv",
            mime = "text/csv"
        )
