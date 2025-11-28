# Missing Data Handling
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer

# load dataset
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
df['diagnosis'] = df['target'].map({0: "Malignant", 1: "Benign"})

# simulate missingness
np.random.seed(42)
for col in ['mean radius', 'mean area', 'worst concavity']:
    df.loc[df.sample(frac=0.05).index, col] = np.nan # 5% missing values
df.loc[df.sample(frac=0.02).index, 'diagnosis'] = np.nan 

''' # summarize missing data
missing_summary = df.isnull().sum() * 100 / len(df)
print(f"\nPercentage of missing values:")
print(f"{missing_summary[missing_summary > 0].sort_values(ascending=False)}")
'''

''' # Handling missing data woth drop
    #safe only if missingness is small and random (MCAR)
# drop rows with any missing values
df_drop_rows = df.dropna()
# drop columns with any missing values
df_drop_cols = df.dropna(axis=1)
print(f"Shape after dropping rows: {df_drop_rows.shape}")
print(f"Shape after dropping columns: {df_drop_cols.shape}")
'''

''' # Handling missing data with Simple Imputation
df_impute = df.copy()
# Numeric feature: fill with mean or median
df_impute['mean radius'] = df_impute['mean radius'].fillna(df_impute['mean radius'].mean())
df_impute['mean area'] = df_impute['mean area'].fillna(df_impute['mean area'].median())
df_impute['worst concavity'] = df_impute['worst concavity'].fillna(df_impute['worst concavity'].mean())
# Categorical features: fill with mode
df_impute['diagnosis'] = df['diagnosis'].fillna(df_impute['diagnosis'].mode()[0])
print("\nMissing values after imputation:")
print(df_impute.isnull().sum().sort_values(ascending=False).head())
'''

'''# Flagging Missingness
    # add flag columns to preserve missingness info
df['mean_radius_missing'] = df['mean radius'].isnull().astype(int)
df['mean_area_missing'] = df['mean area'].isnull().astype(int)
df['worst_concavity_missing'] = df['worst concavity'].isnull().astype(int)
df['diagnosis_missing'] = df['diagnosis'].isnull().astype(int)
print("\nFlag columns added:")
print(df[['mean_radius_missing', 'mean_area_missing', 'worst_concavity_missing', 'diagnosis_missing']].head())
print(len(df['mean_radius_missing']))
'''

# Data type fixes
data = {
    'Patient_ID': ['101', '102', '103', '104'],   # numeric-looking strings
    'Age': ['45', 'NaN', '50', '62'],             # strings with "NaN"
    'Visit_Date': ['2025-11-28', '28/11/2025', 'Nov 28, 2025', '2025/11/28'],  # mixed date formats
    'Diagnosis': ['Malignant', 'Benign', 'Malignant', 'Benign'],  # categorical 
    'Weight': ['70kg', '65kg', 'NaN', '80kg']       
}
df_fixing = pd.DataFrame(data)
    
'''    # fix numeric-looking strings (patient_ID and age to numeric)
df_fixing['Patient_ID'] = pd.to_numeric(df_fixing['Patient_ID'], errors='coerce') # coerce turns invalid entries like "NaN" to actual NaN
df_fixing['Age'] = pd.to_numeric(df_fixing['Age'], errors='coerce')
print(df_fixing.dtypes)
'''
'''    # fix dates (convert Visit_Date to datetime)
df_fixing['Visit_Date'] = pd.to_datetime(df_fixing['Visit_Date'], errors="coerce", dayfirst=True, infer_datetime_format=True) # format='%Y-%m-%d' or (dayfirst=True, infer_datetime_format=True)
print(df_fixing.dtypes)
print(df_fixing['Visit_Date'])
'''
'''    # fixing categorical Data (diagnosis to category type)
df_fixing['diagnosis'] = df['diagnosis'].astype('category')
print(df_fixing.dtypes)
print(df_fixing['diagnosis'].cat.codes) # encoded values
'''
'''    # fixing different formats of dates
dates = ['2025-11-28', '28/11/2025', 'Nov 28, 2025', '2025/11/28']
df_dates = pd.DataFrame({'Visit_date': dates})
formats = ['%Y-%m-%d', '%d/%m/%Y', '%b %d, %Y', '%Y/%m/%d'] # b = Abbreviated month name eg Jan, Feb, Mar etc
    # use more formats if you need them
def parse_date(x):
    for format in formats:
        try: # if this format works, great! Else, we catch error and continue
            return pd.to_datetime(x, format=format)
        except: # run this if occur in try: block
            continue
    return pd.Nat # Not-a-Time in pd; NaN in python
df_dates['Visit_date'] = df_dates['Visit_date'].apply(parse_date)
print(df_dates)
'''
''' # fix numeric-looking strings with extra characters
df_string = pd.DataFrame(data)
print(f"Original: {df_string.dtypes}")
    
    # Patient_ID and Age to numeric
df_string['Patient_ID'] = pd.to_numeric(df_string['Patient_ID'], errors='coerce')
df_string['Age'] = pd.to_numeric(df_string['Age'], errors='coerce')
    # Clean weight column (remove 'Kg' then convert)
df_string['Weight'] = df_string['Weight'].str.replace('Kg', '', regex=False)
    # replace 'Kg' by .str.replace('Kg', '', regex=False); regex -> do simple replacement, not using regex
df_string['Weight'] = pd.to_numeric(df_string['Weight'], errors='coerce')
print(f"\nCleaned dtypes:\n {df_string.dtypes}")
'''

# Categorical Encoding
print(df_fixing['Diagnosis'])
df_fixing['Diagnosis'] = df_fixing['Diagnosis'].astype('category')
print(df_fixing['Diagnosis'].cat.codes) # label encoding
print(pd.get_dummies(df_fixing['Diagnosis'])) # one-hot encoding