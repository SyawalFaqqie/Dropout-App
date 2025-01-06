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

    if "predicted_data" not in st.session_state:
        st.session_state.predicted_data = None


    #Data Prediction
    if st.button("Predict"):
        try:

            predictions = model.predict(data[features])
            probabilities = model.predict_proba(data[features])[:,1]

            st.session_state.predicted_data = data.copy()
            st.session_state.predicted_data["Dropout_Prediction"] = [
                "Yes" if pred == 1 else "No" for pred in predictions
            ]
            st.session_state.predicted_data["Risk_Score"] = probabilities.round(2)

            st.success("Predictions completed!")
            
        except Exception as e:
            st.error(f"An error occured during prediction: {e}")

if st.session_state.predicted_data is not None:
    # Provide filtering options for the user
    filter_option = st.selectbox(
        "Select predicted value to view:",
        options=["All", "Yes", "No"],
        index=0,
        key="filter_option"
    )
    scholarship_filter = st.selectbox("Select Scholarship Holder", options = ["All",1,0])
    displaced_filter = st.selectbox("Select Displaced Status", options=["All",1,0])
    

    # Filter data based on selection
    filtered_data = st.session_state.predicted_data.copy()
    if filter_option != "All":
        filtered_data = st.session_state.predicted_data[
            st.session_state.predicted_data["Dropout_Prediction"] == filter_option]
    if scholarship_filter != "All":
        filtered_data = st.session_state.predicted_data[
            st.session_state.predicted_data["Scholarship holder"] == scholarship_filter]
    if displaced_filter != "All":
        filtered_data = st.session_state.predicted_data[
            st.session_state.predicted_data["Displaced"] == displaced_filter]

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

    st.write(f"### Showing {filter_option} Records:")
    if not filtered_data.empty:
        st.dataframe(filtered_data)

        # Download filtered data
        filtered_csv = filtered_data.to_csv(index=False)
        st.download_button(
            label=f"Download filtered Records as CSV",
            data=filtered_csv,
            file_name=f"{filter_option.lower()}_records.csv",
            mime="text/csv"
        )
    else:
        st.write(f"No records found for '{filter_option}'.")      
        
   
