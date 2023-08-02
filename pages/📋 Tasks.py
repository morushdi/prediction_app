import streamlit as st
import pandas as pd
import numpy as np
import time
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score, mean_absolute_error, max_error
from math import sqrt


# import custom modules
from lib.utils import plotData_prediction
from lib.utils import quality_metrics_single
from lib.utils import quality_metrics_multi
from lib.lr import LR
from lib.nn import NN
# ==================== ##
## **** load logos **** ##
# ==================== ##
st.sidebar.image('data/Hitachi-logo.png', use_column_width=True)
st.sidebar.image('data/KU_logo.png', use_column_width=True)



# ==================== ##
## **** Choose Task **** ##
# ==================== ##

option = st.selectbox(
    'Which problem would you like to solve?',
    ('None', 'Single-output', 'Multi-output'))

## ==================== ##
## **** Single-output **** ##
## ==================== ##

if option == 'Single-output':
    st.title('Single-output Model')
    st.write('Here you can use the single-output model')
    st.write('This model is used to predict the **hub wind speed** based on the following inputs:')
    st.markdown("""
                - Generator speed
                - Pitch Angles
                - Power output
                - Rotor azimuth angle
                - Yaw misalignment
                - MXB/MYB: Blade root bending moment (in-plane & out-of-plane)
                """)
    
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df = df.dropna()
        
        

        # Check if header row matches expected header
        expected_header = set(["Electrical power", "Yaw misalignment", "Mean pitch angle", "Rotor azimuth angle", "Generator speed", 
                           "Blade root 1 Mx", "Blade root 1 My", "Blade root 2 Mx", "Blade root 2 My", "Blade root 3 Mx", "Blade root 3 My"])
        header = set(df.columns)
        if not expected_header.issubset(header):
            st.error("Invalid header row in the uploaded CSV file. Please make sure the header matches the expected format.")
            st.stop()  # Stop further execution of the code in this script.


        if df is not None:
            st.success('File loaded successfully')
            if st.checkbox('Show raw data'):
                st.subheader('Raw data')
                st.write(df)
            if st.checkbox('Show data description'):
                st.subheader('Data description')
                st.write(df.describe().transpose())
            # --------------------- ##
            ## **** Prediction **** ##
            # --------------------- ##
            st.subheader('Prediction')
            lr = LR('models/lr-single.pkl')
            st.write("Hub wind speed magnitude prediction using linear regression") 
            features = ["Electrical power", "Yaw misalignment", "Mean pitch angle", "Rotor azimuth angle", "Generator speed", 
                        "Blade root 1 Mx", "Blade root 1 My", "Blade root 2 Mx", "Blade root 2 My", "Blade root 3 Mx", "Blade root 3 My", "previous Hub wind speed magnitude"]
            x = pd.DataFrame(df, columns=features)
            predictions = lr.predict(x)
            # ---------------------
            # *** Show progress bar *** #
            # ---------------------
            # predictions = []
            # progress_bar = st.progress(0)
            # status_text = "Predicting..."
            # for i in range(len(df)):
            #     # Simulate prediction step
            #     time.sleep(0.00002)
            #     prediction = lr.predict([x.iloc[i]])
            #     predictions.append(prediction[0])

            #     # Update progress bar
            #     progress_bar.progress((i + 1) / len(df))

            # # Clear progress bar
            # progress_bar.empty()

          

            df['predictions'] = predictions
            st.dataframe(df)
            # append predictions to dataframe
            prediction_csv = df.to_csv('results/single_output.csv')
            
            df = pd.read_csv("results/single_output.csv")
            @st.cache_data
            def convert_df(df):
                    return df.to_csv(index=False).encode('utf-8')
                
            csv = convert_df(df)
            st.download_button(
            "Press to Download",
            csv,
            "results/single_output.csv",
            "text/csv",
            key='download-csv'
            )
            # ------------------------------ ##
            ## **** True Data Available **** ##
            # ------------------------------ ##
            if st.checkbox('I have true data to compare'):
                # ------------------------ ##
                ## **** Visualization **** ##
                # -------------------------##
                st.subheader('Visualization')
                st.write('Here you can compare the true data with the predicted data')

                y_true = df['Hub wind speed magnitude']
                y_pred = df['predictions']
                label = 'Hub wind speed magnitude'
                set = 'Validation'

                plotData_prediction(y_true, y_pred, label, set)
                # -------------------------- ##
                ## **** Quality Metrics **** ##
                # -------------------------- ##
                quality_metrics_single(y_true, y_pred)

## ====================== ##
## **** Multi-output **** ##
## ====================== ##
if option == 'Multi-output':
    st.title('Multi-output Model')
    st.write('Here you can use the multi-output model')
    st.write('This model is used to predict the **WT loads** ')
    st.markdown("""
                - Outputs:
                    - MXN: Rotor Torque 
                    - MYN/MZN: Rotor vertical/horizontal bending 
                    - FXN: Rotor axial force
                - Inputs:
                    - Operation data: Generator speed, Pitch Angles, Power output, Rotor azimuth angle, Yaw misalignment, nacelle acceleration
                    - MXB/MYB: Blade root bending moment (in-plane & out-of-plane)
                    - MXF/MYF: Tower base bending moment
                    - MXT/MYT: Tower top bending moment
                """)
    
    
    model = st.selectbox(
    'Which model would you like to use?',
    ('None', 'Neural Network', 'Linear Regression'))
    if model == 'None':
         pass
    # ========================= ##
    ## **** Neural Network **** ##
    # ========================= ##
    elif model == 'Neural Network':
        st.write('Neural Network')
        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            df = df.dropna()

            # Check if header row contain expected header
            expected_header = set(["Electrical power", "Yaw misalignment", "Mean pitch angle", "Rotor azimuth angle", "Generator speed", 
                               "Blade root 1 Mx", "Blade root 1 My", "Blade root 2 Mx", "Blade root 2 My", "Blade root 3 Mx", "Blade root 3 My", 
                               "Nacelle roll acceleration", "Nacelle yaw acceleration" , "Nacelle nod acceleration",
                               "MXT(Tower Bottom)", "MYT(Tower Bottom)", "MZT(Tower Bottom)", 
                               "MXT(Tower Top)", "MYT(Tower Top)", "MZT(Tower Top)"])
            header = set(df.columns)
            if not expected_header.issubset(header):
                st.error("Invalid header row in the uploaded CSV file. Please make sure the header matches the expected format.")
                st.stop()  # Stop further execution of the code


            if df is not None:
                st.success('File loaded successfully')
                if st.checkbox('Show raw data'):
                    st.subheader('Raw data')
                    st.write(df)
                if st.checkbox('Show data description'):
                    st.subheader('Data description')
                    st.write(df.describe().transpose())
                # --------------------- ##
                ## **** Prediction **** ##
                # --------------------- ##
                st.subheader('Prediction')
                nn = NN('models/NN-multi-tuned.h5')
                st.write("Predictions using Neural Network")
                features = ["Electrical power", "Yaw misalignment", "Mean pitch angle", "Rotor azimuth angle", "Generator speed", 
                            "Blade root 1 Mx", "Blade root 1 My", "Blade root 2 Mx", "Blade root 2 My", "Blade root 3 Mx", "Blade root 3 My", 
                            "Nacelle roll acceleration", "Nacelle yaw acceleration" , "Nacelle nod acceleration",
                            "MXT(Tower Bottom)", "MYT(Tower Bottom)", "MZT(Tower Bottom)", 
                            "MXT(Tower Top)", "MYT(Tower Top)", "MZT(Tower Top)"]
                x = pd.DataFrame(df, columns=features)
                predictions = nn.predict(x)
                # ==================== ##

                df['Stationary hub Mx predicted'] = predictions[:,0]
                df['Stationary hub My predicted'] = predictions[:,1]
                df['Stationary hub Mz predicted'] = predictions[:,2]
                df['Stationary hub Fx predicted'] = predictions[:,3]

                st.dataframe(df)
                # append predictions to dataframe
                prediction_csv = df.to_csv('results/multi_output.csv')
                
                df = pd.read_csv("results/multi_output.csv")
                @st.cache_data
                def convert_df(df):
                        return df.to_csv(index=False).encode('utf-8')
                    
                csv = convert_df(df)
                st.download_button(
                "Press to Download",
                csv,
                "results/multi_output.csv",
                "text/csv",
                key='download-csv'
                )
                # ==================== ##
                # ------------------------------ ##
                ## **** True Data Available **** ##
                # ------------------------------ ##
                if st.checkbox('I have true data to compare'):
                    # ------------------------ ##
                    ## **** Visualization **** ##
                    # -------------------------##
                    st.subheader('Visualization')
                    st.write('Here you can compare the true data with the predicted data')

                    visualization = st.selectbox(
                    'Which parameter would you like to visualize?',
                    ('None', 'Stationary hub Mx', 'Stationary hub My', 'Stationary hub Mz', 'Stationary hub Fx'))
                    if model == 'None':
                        pass
                    elif visualization == 'Stationary hub Mx':
                        y_true = df['Stationary hub Mx']
                        y_pred = df['Stationary hub Mx predicted']
                        label = 'Stationary hub Mx'
                        set = 'Validation'
                        plotData_prediction(y_true, y_pred, label, set)
                        quality_metrics_multi(y_true, y_pred)

                    elif visualization == 'Stationary hub My':
                        y_true = df['Stationary hub My']
                        y_pred = df['Stationary hub My predicted']
                        label = 'Stationary hub My'
                        set = 'Validation'
                        plotData_prediction(y_true, y_pred, label, set)
                        quality_metrics_multi(y_true, y_pred)

                    elif visualization == 'Stationary hub Mz':
                        y_true = df['Stationary hub Mz']
                        y_pred = df['Stationary hub Mz predicted']
                        label = 'Stationary hub Mz'
                        set = 'Validation'
                        plotData_prediction(y_true, y_pred, label, set)
                        quality_metrics_multi(y_true, y_pred)

                    elif visualization == 'Stationary hub Fx':
                        y_true = df['Stationary hub Fx']
                        y_pred = df['Stationary hub Fx predicted']
                        label = 'Stationary hub Fx'
                        set = 'Validation'
                        plotData_prediction(y_true, y_pred, label, set)
                        quality_metrics_multi(y_true, y_pred)
    # ========================= ##
    ## **** Linear Regression **** ##
    # ========================= ##
    elif model == 'Linear Regression':
        st.write('Linear Regression')
        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            df = df.dropna()


            # Check if header row contain expected header
            expected_header = set(["Electrical power", "Yaw misalignment", "Mean pitch angle", "Rotor azimuth angle", "Generator speed", 
                               "Blade root 1 Mx", "Blade root 1 My", "Blade root 2 Mx", "Blade root 2 My", "Blade root 3 Mx", "Blade root 3 My", 
                               "Nacelle roll acceleration", "Nacelle yaw acceleration" , "Nacelle nod acceleration",
                               "MXT(Tower Bottom)", "MYT(Tower Bottom)", "MZT(Tower Bottom)", 
                               "MXT(Tower Top)", "MYT(Tower Top)", "MZT(Tower Top)"])
            header = set(df.columns)
            if not expected_header.issubset(header):
                st.error("Invalid header row in the uploaded CSV file. Please make sure the header matches the expected format.")
                st.stop()  # Stop further execution of the code


            if df is not None:
                st.success('File loaded successfully')
                if st.checkbox('Show raw data'):
                    st.subheader('Raw data')
                    st.write(df)
                if st.checkbox('Show data description'):
                    st.subheader('Data description')
                    st.write(df.describe().transpose())
                # --------------------- ##
                ## **** Prediction **** ##
                # --------------------- ##
                st.subheader('Prediction')
                lr = LR('models/lr-multi.pkl')
                st.write("Predictions using Linear Regression model")
                features = ["Electrical power", "Yaw misalignment", "Mean pitch angle", "Rotor azimuth angle", "Generator speed", 
                            "Blade root 1 Mx", "Blade root 1 My", "Blade root 2 Mx", "Blade root 2 My", "Blade root 3 Mx", "Blade root 3 My", 
                            "Nacelle roll acceleration", "Nacelle yaw acceleration" , "Nacelle nod acceleration",
                            "MXT(Tower Bottom)", "MYT(Tower Bottom)", "MZT(Tower Bottom)", 
                            "MXT(Tower Top)", "MYT(Tower Top)", "MZT(Tower Top)"]
                x = pd.DataFrame(df, columns=features)
                predictions = lr.predict(x)

                df['Stationary hub Mx predicted'] = predictions[:,0]
                df['Stationary hub My predicted'] = predictions[:,1]
                df['Stationary hub Mz predicted'] = predictions[:,2]
                df['Stationary hub Fx predicted'] = predictions[:,3]

                st.dataframe(df)
                # append predictions to dataframe
                prediction_csv = df.to_csv('results/multi_output.csv')
                
                df = pd.read_csv("results/multi_output.csv")
                @st.cache_data
                def convert_df(df):
                        return df.to_csv(index=False).encode('utf-8')
                    
                csv = convert_df(df)
                st.download_button(
                "Press to Download",
                csv,
                "results/multi_output.csv",
                "text/csv",
                key='download-csv'
                )
                # ==================== ##
                # ------------------------------ ##
                ## **** True Data Available **** ##
                # ------------------------------ ##
                if st.checkbox('I have true data to compare'):
                    # ------------------------ ##
                    ## **** Visualization **** ##
                    # -------------------------##
                    st.subheader('Visualization')
                    st.write('Here you can compare the true data with the predicted data')

                    visualization = st.selectbox(
                    'Which parameter would you like to visualize?',
                    ('None', 'Stationary hub Mx', 'Stationary hub My', 'Stationary hub Mz', 'Stationary hub Fx'))
                    if model == 'None':
                        pass
                    elif visualization == 'Stationary hub Mx':
                        y_true = df['Stationary hub Mx']
                        y_pred = df['Stationary hub Mx predicted']
                        label = 'Stationary hub Mx'
                        set = 'Validation'
                        plotData_prediction(y_true, y_pred, label, set)
                        quality_metrics_multi(y_true, y_pred)

                    elif visualization == 'Stationary hub My':
                        y_true = df['Stationary hub My']
                        y_pred = df['Stationary hub My predicted']
                        label = 'Stationary hub My'
                        set = 'Validation'
                        plotData_prediction(y_true, y_pred, label, set)
                        quality_metrics_multi(y_true, y_pred)

                    elif visualization == 'Stationary hub Mz':
                        y_true = df['Stationary hub Mz']
                        y_pred = df['Stationary hub Mz predicted']
                        label = 'Stationary hub Mz'
                        set = 'Validation'
                        plotData_prediction(y_true, y_pred, label, set)
                        quality_metrics_multi(y_true, y_pred)

                    elif visualization == 'Stationary hub Fx':
                        y_true = df['Stationary hub Fx']
                        y_pred = df['Stationary hub Fx predicted']
                        label = 'Stationary hub Fx'
                        set = 'Validation'
                        plotData_prediction(y_true, y_pred, label, set)
                        quality_metrics_multi(y_true, y_pred)
         

