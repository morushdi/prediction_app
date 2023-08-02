
# About

Prediction of WS and WT loads using ML

- **Data Analysis**: Represents overview of the data used in this app.
    - Note: The data used in this app is not the original data, but a subset of it. The original data is not publicly available.
- **Tasks**:
    - Task 1 [single-output model]➡︎ Predict hub wind speed (1 output) based on 12 input features.
    - Task 2 [multi-output model] ➡︎ Predict WT loads (4 outputs) based on 20 input features.

# Reproduction 

To run the streamlit app, or run the repo code, it is advised to activate a python or a conda environment.

1. Navigate to the project folder:

2. Create a conda environment from the provided environment file
```bash 
conda env create -f environment.yml
```

3. Activate the new environment: 
```bash
conda activate myenv
```
4. Verify that the new environment was installed correctly:
```bash
conda env list
```
5. To run the streamlit app:
```bash
streamlit run Home.py
```



