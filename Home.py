import streamlit as st
import pandas as pd


# import custom modules
from lib import lr
from lib.utils import process_csv_files


# =============================== #
st.header('KU-Hitachi App 💻')
st.title('Prediction of WS and WT loads using ML')
# =============================== #

# ==================== ##
## **** load logos **** ##
# ==================== ##
st.sidebar.image('data/Hitachi-logo.png', use_column_width=True)
st.sidebar.image('data/KU_logo.png', use_column_width=True)

# ==================== ##
## **** Authors **** ##
# ==================== ##
st.subheader('Authors ✍🏻')
st.markdown("""
            - 👨🏻‍💻 **App development**: Mostafa A. Rushdi, _RIAM, Kyushu University, Japan_
            - 📈 **Data Collection**: Soichiro Kiyoki, _Power Busniess Unit, Hitachi Co., Japan_
            - 🧑🏻‍🏫 **Supervision**: Prof. Shigeo Yoshida, _RIAM, Kyushu University, Japan_
            """)

st.divider()
# =============================== #

# ==================== ##
## **** Problem Illustration **** ##
# ==================== ##
st.subheader('Problem Illustration ')
st.markdown("""
            - **System**: Wind Turbine (What type of WT? / What is the capacity of the WT?)
            - **Goal** 🎯: Wind and loads evaluation/estimation
            - **Importance**: Fatigue loads of WT are found to be sensitive to wind conditions of the site. But the detailed wind conditions as wind shear, inclinations are not always measured at site.
            - **Solution**: Using ML to predict wind and loads from the available data.
            - **Data**: The data used in this app is generated using a simulation.
                - The simulation is done using a wind turbine model in FAST software.
                - The simulation is done for 2 different wind conditions (wind speed = 8 m/s and 14 m/s), with average wind sheer = 0.2
                - The simulation is done with sampling = 20 Hz, Duration = 10 min/case
            """)
st.divider()
# =============================== #

# ==================== ##
## **** App Illustration **** ##
# ==================== ##
st.subheader('App Illustration ⚙️')
st.markdown("""
            - **Data Analysis**: Represents overview of the data used in this app.
            - **Tasks**:
                - Task 1 [single-output model]➡︎ Predict hub wind speed (1 output) based on 12 input features.
                - Task 2 [multi-output model] ➡︎ Predict WT loads (4 outputs) based on 20 input features.
            """)

st.divider()
# =============================== #












