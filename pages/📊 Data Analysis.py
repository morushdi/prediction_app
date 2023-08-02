import streamlit as st
import pandas as pd


# import custom modules
from lib.utils import process_csv_files


# ==================== ##
## **** Figure Style **** ##
# ==================== ##
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

# ==================== ##
## **** load logos **** ##
# ==================== ##
st.sidebar.image('data/Hitachi-logo.png', use_column_width=True)
st.sidebar.image('data/KU_logo.png', use_column_width=True)

# ==================== ##
## **** Title **** ##
# ==================== ##
st.title('Exploratory Data Analysis')

# ==================== ##
## **** load data **** ##
# ==================== ##
Data_dir = 'data'

# path1 = f"{Data_dir}/f307v080"
# df1 = process_csv_files(path1)

# path2 = f"{Data_dir}/f307v140"
# df2 = process_csv_files(path2)

# df = pd.concat([df1, df2], ignore_index=True, sort=False)

df = process_csv_files(Data_dir)
#df = pd.read_csv('data/df.csv')
# =============================== #


# ==================== ##
## **** Dataframe **** ##
# ==================== ##
st.subheader('Data Overview')
st.write(df.head())

st.subheader('Data Description/Statistics')
st.write(df.describe().transpose())


# ==================== ##
## **** PCC Heatmap **** ##
# ==================== ##
st.subheader('PCC Heatmap')
corr_matrix = df.corr()

mpl.rcParams['axes.unicode_minus'] = False
with plt.style.context(['science','ieee','no-latex', 'std-colors','grid']):
    plt.rc('font', family='Gulliver')
    
    fig = plt.figure(figsize=(50,50))
    ax = sns.heatmap(corr_matrix, vmax=1, vmin = -1, center = 0, square=True, annot=True, annot_kws={"size":11}, fmt='.2f', cmap ='RdBu', cbar_kws={"shrink": .5}, robust=True)
    #plt.title('Correlation matrix between the features', fontsize=20)

    ax.set_xticklabels(ax.get_xticklabels(), rotation = 90, fontsize = 12)
    ax.set_yticklabels(ax.get_yticklabels(), rotation = 0, fontsize = 12)
    
    cax = plt.gcf().axes[-1]
    cax.tick_params(labelsize=12)
    

    plt.tight_layout()

st.pyplot(fig)

# ==================== ##
## **** Joint Plots **** ##
# ==================== ##
with plt.style.context(['science','ieee','no-latex', 'std-colors','grid']):
    plt.rc('font', family='Gulliver') 

    h = sns.jointplot(x = 'previous Hub wind speed magnitude', y = 'Hub wind speed magnitude', data = df, kind="reg", height=5, ratio=3, marginal_ticks=True, line_kws={'color': 'red'})

    h.ax_joint.set_ylabel("Hub wind speed magnitude")
    h.ax_joint.set_xlabel("previous Hub wind speed magnitude")

    plt.tight_layout()

st.pyplot(h)





