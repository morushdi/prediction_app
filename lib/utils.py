import os
import pandas as pd
import numpy as np
import streamlit as st

from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score, mean_absolute_error, max_error
from math import sqrt

# ======================= ##
## **** Figure Style **** ##
# ======================= ##
import seaborn as sns
import matplotlib.pyplot as plt
Fonts = '/Users/mostafarushdi/Downloads/Fonts'

# import scienceplots # => with version 2.0.0
plt.style.reload_library()
plt.style.use('science')


import matplotlib as mpl
mpl.rcParams['axes.unicode_minus'] = False

# upload  ttf file of the font
mpl.font_manager.fontManager.addfont(f"{Fonts}/Gulliver-Regular.ttf") # otf worked okay but there was some kind of error/warning so its better to just use ttf
mpl.rc('font', family='Gulliver')


plt.rcParams.update({
    "font.family": "serif",   # specify font family here
    "font.serif": ["Gulliver"],  # specify font here
    "font.size":11})          # specify font size here
# =============================== #


## ========================= ##
## **** Loading Data fn **** ##
## ========================= ##

def process_csv_files(directory):
    file_extension = '.csv'
    csv_file_list = []
    for root, dirs, files in os.walk(directory):
        for name in files:
            if name.endswith(file_extension):
                file_path = os.path.join(root, name)
                df_file = pd.read_csv(file_path)
                df_file = df_file.sort_values('Time from start of output')
                df_file['previous Hub wind speed magnitude'] = df_file['Hub wind speed magnitude'].shift(1)
                csv_file_list.append(df_file)

    return pd.concat(csv_file_list, ignore_index=True)





## ============================== ##
## **** Visualize Results fn **** ##
## ============================== ##

def plotData_prediction(y_true, y_pred, label, set):

  y = y_true 
  yp = y_pred 
  diff = np.abs(y-yp)
  error = yp-y

## ============== Plotting ================
## ------ True vs Prediction plot -----
  with plt.style.context(['science','ieee','no-latex', 'std-colors','grid']):
    plt.rc('font', family='Gulliver') 
    fig, ax = plt.subplots(figsize=(4,4))

    #ax = plt.axes(aspect='equal')
    sns.regplot(x=y, y=yp, line_kws={'color': 'red'}, label='regression')

    plt.xlabel('True Values (' + label +  ') [' + set + ' set]')
    plt.ylabel('Predictions (' + label +  ') [' + set + ' set]')
    #plt.title('OrientAngle')

    line_45 = np.linspace(*ax.get_xlim())
    ax.plot(line_45, line_45, color='g', label="$45^o$")

    plt.legend(loc='lower right')
    # plt.xlim(0, 3.7)
    # plt.ylim(0, 3.7)

    plt.tight_layout()
    st.pyplot(fig)
    st.caption("True vs Prediction plot. The closer the points to the 45-degree line, the better the prediction is.")

## --------------------------------------------
## ------ True-Prediction time history -------
  with plt.style.context(['science','ieee','no-latex', 'std-colors','grid']):
    plt.rc('font', family='Gulliver') 
    fig, ax = plt.subplots(figsize=(10,5))

    #fig= plt.figure(figsize=(25,10))
    plt.plot(y, label="True")
    plt.plot(yp, label="Prediction", color='orange')

    plt.legend(title= ' Validation Set', loc='lower right')
    #plt.title("True & predicted [Train]")
    plt.tight_layout()
    st.pyplot(fig)
    st.caption("True & predicted time history.")
    

## --------------------------------------------

  # Error = pd.DataFrame({"True":y, "Predicted":yp, "Error":error})
  # Error.to_csv(f"{Figures_Dir}/Error_Train.csv")

## ------- Error histogram plot --------
  with plt.style.context(['science','ieee','no-latex', 'std-colors','grid']):
    plt.rc('font', family='Gulliver') 
    fig, ax = plt.subplots(figsize=(4,4))

    plt.hist(error, bins = 1000)

    plt.xlabel('Prediction Error (' + label +  ') [' + set + ' set]')
    plt.ylabel("Count")
    plt.xlim(np.min(error), np.max(error))


    plt.tight_layout()
    st.pyplot(fig)
    st.caption("Prediction Error histogram plot.")

## --------------------------------------------  
  with plt.style.context(['science','ieee','no-latex', 'std-colors','grid']):
    plt.rc('font', family='Gulliver') 
    fig, ax = plt.subplots(figsize=(6,4))

    plt.scatter(x=y, y=diff)

    plt.xlabel('True values (' + label +  ') [' + set + ' set]')
    plt.ylabel('|Prediction Error| (' + label +  ') [' + set + ' set]')
    #plt.xlim(np.min(error), np.max(error))

    plt.tight_layout()
    st.pyplot(fig)
    st.caption("True values vs |Prediction Error| plot.")


## ============== Stats ================
  # npoints = ((diff > 0.5).sum())*100/diff.size
  # npoints_1 = ((diff > 1.0).sum())*100/diff.size
  # npoints_2 = ((diff > 1.5).sum())*100/diff.size

  # lines = ['Mean Error in WindSpeed: ' + str(np.mean(diff)) + ' m/s',
  #          'RMSE Error in WindSpeed: ' + str(np.sqrt(np.mean(diff*diff))) + ' m/s',
  #          'Max Error in WindSpeed: ' + str(np.max(diff)) + ' m/s',
  #          f' {npoints:4.2f} % of points have WindSpeed > 0.5 m/s ',
  #          f' {npoints_1:4.2f} % of points have WindSpeed > 1.0 m/s ',
  #          f' {npoints_2:4.2f} % of points have WindSpeed > 1.5 m/s ',
  #          'Min Error in WindSpeed ' + str(np.min(diff)) + ' m/s']

  # with open(f"{Figures_Dir}/Stats_Train.txt", 'w') as f:
  #   for line in lines:
  #       f.write(line)
  #       f.write('\n')





# ============================= ##
## **** Quality Metrics fn **** ##
# ============================= ##
def quality_metrics_single(y_true, y_pred):
  st.subheader('Quality Metrics')
  st.write('Here you can see the quality metrics of the model')
  st.write('The quality metrics are:')

  # Calculate quality metrics
  mse = mean_squared_error(y_true, y_pred)
  rmse = sqrt(mse)
  c_d = r2_score(y_true, y_pred)
  e_max = max_error(y_true, y_pred)
  exp_var = explained_variance_score(y_true, y_pred, multioutput='uniform_average')
  mae = mean_absolute_error(y_true, y_pred)

  # Display quality metrics as bullet points
  st.markdown("""
              - MSE: %.5f
              - RMSE: %.5f
              - C_D: %.5f
              - E_max: %.5f
              - Exp_Var: %.5f
              - MAE: %.5f
              """ % (mse, rmse, c_d, e_max, exp_var, mae))
  
 # ============================= ##
def quality_metrics_multi(y_true, y_pred):
  st.subheader('Quality Metrics')
  st.write('Here you can see the quality metrics of the model')
  st.write('The quality metrics are:')

  # Calculate quality metrics
  mse = mean_squared_error(y_true, y_pred)
  rmse = sqrt(mse)
  c_d = r2_score(y_true, y_pred)
  exp_var = explained_variance_score(y_true, y_pred, multioutput='uniform_average')
  mae = mean_absolute_error(y_true, y_pred)

  # Display quality metrics as bullet points
  st.markdown("""
              - MSE: %.5f
              - RMSE: %.5f
              - C_D: %.5f
              - Exp_Var: %.5f
              - MAE: %.5f
              """ % (mse, rmse, c_d, exp_var, mae)) 