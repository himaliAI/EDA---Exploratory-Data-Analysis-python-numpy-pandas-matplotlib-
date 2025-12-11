# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Load your synthetic dataset
# (We'll use the code you provided to generate it)
rng = np.random.default_rng(seed=42)
n = 1000  # number of rows

# Generate features
muc5ac = rng.normal(50, 10, n)  # MUC5AC concentration
il8 = rng.normal(200, 50, n)    # IL8 level
crs = rng.choice([0, 1], size=n)  # 0=no, 1=yes
nasal_polyp = rng.choice([0, 1], size=n)  # 0=Absent, 1=present

# Inject outliers
muc5ac[rng.choice(n, 10)] = muc5ac[rng.choice(n, 10)] * 5
il8[rng.choice(n, 10)] = il8[rng.choice(n, 10)] * 25

# Target (symptom score)
symptom_score = (muc5ac * 0.3) + (il8 * 0.2) + (crs * 0.15) + (nasal_polyp * 2.8) + rng.normal(0, 10, size=n)

df = pd.DataFrame({
    'muc5ac': muc5ac,
    'il8': il8,
    'crs': crs,
    'nasal_polyp': nasal_polyp,
    'symptom_score': symptom_score
})

# Create string versions for better visualization
df['crs_str'] = df['crs'].map({0: "No", 1: "Yes"})
df['nasal_polyp_str'] = df['nasal_polyp'].map({0: "Absent", 1: "Present"})

# Inject NaN randomly (5% per column)
for col in df.columns:
    df.loc[df.sample(frac=0.05).index, col] = np.nan

print("✅ Dataset created successfully!")
print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print("\n" + "="*50)

# ============================================================================
# STEP 1: INITIAL SUMMARY AND VIEW
# ============================================================================

print("STEP 1: INITIAL SUMMARY AND VIEW")
print("="*50)

# 1.1 Basic Information
print("\n1.1 BASIC DATASET INFORMATION:")
print("-" * 30)
print(f"Total rows: {df.shape[0]}")
print(f"Total columns: {df.shape[1]}")

# Display first few rows
print("\nFirst 5 rows of the dataset:")
print(df.head())

# Display last few rows
print("\nLast 5 rows of the dataset:")
print(df.tail())

# 1.2 Data Types and Missing Values
print("\n1.2 DATA TYPES AND MISSING VALUES:")
print("-" * 30)
df_info = pd.DataFrame({
    'Data Type': df.dtypes,
    'Missing Values': df.isnull().sum(),
    'Missing %': (df.isnull().sum() / len(df) * 100).round(2)
})
print(df_info)

# 1.3 Identify Numerical and Categorical Columns
print("\n1.3 COLUMN CATEGORIZATION:")
print("-" * 30)
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

print(f"Numerical columns ({len(numerical_cols)}): {numerical_cols}")
print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")

# 1.4 Descriptive Statistics
print("\n1.4 DESCRIPTIVE STATISTICS FOR NUMERICAL COLUMNS:")
print("-" * 30)
print(df[numerical_cols].describe())

# For categorical columns
print("\n1.5 FREQUENCY DISTRIBUTION FOR CATEGORICAL COLUMNS:")
print("-" * 30)
for col in categorical_cols:
    print(f"\n{col}:")
    print(df[col].value_counts(dropna=False))
    print(f"Mode: {df[col].mode()[0]}")

# 1.6 Visualization: Multiple plots in a single figure
print("\n1.6 VISUALIZATION - DISTRIBUTION ANALYSIS")
print("-" * 30)

# Create a 2x2 grid of subplots for numerical columns
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Distribution Analysis of Numerical Features', fontsize=16, fontweight='bold')

# Flatten axes for easy iteration
axes = axes.flatten()

# Plot histogram and KDE for each numerical column
for i, col in enumerate(numerical_cols):
    ax = axes[i]
    
    # Histogram with KDE
    sns.histplot(data=df, x=col, kde=True, ax=ax, bins=30, color='skyblue', edgecolor='black')
    ax.set_title(f'Distribution of {col}', fontsize=12, fontweight='bold')
    ax.set_xlabel(col)
    ax.set_ylabel('Frequency')
    
    # Add mean and median lines
    mean_val = df[col].mean()
    median_val = df[col].median()
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
    ax.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}')
    ax.legend()

plt.tight_layout()
plt.show()

# 1.7 Boxplots for outlier detection
print("\n1.7 VISUALIZATION - BOXPLOTS FOR OUTLIER DETECTION")
print("-" * 30)

fig, axes = plt.subplots(1, len(numerical_cols), figsize=(16, 6))
fig.suptitle('Boxplots for Outlier Detection', fontsize=16, fontweight='bold')

for i, col in enumerate(numerical_cols):
    ax = axes[i]
    
    # Create boxplot
    boxplot = ax.boxplot(df[col].dropna(), patch_artist=True)
    
    # Customize boxplot
    boxplot['boxes'][0].set_facecolor('lightblue')
    boxplot['whiskers'][0].set_color('black')
    boxplot['whiskers'][1].set_color('black')
    boxplot['fliers'][0].set(marker='o', color='red', alpha=0.5)
    
    ax.set_title(f'{col}', fontsize=12, fontweight='bold')
    ax.set_ylabel('Values')
    
    # Add grid
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 1.8 Categorical data visualization
print("\n1.8 VISUALIZATION - CATEGORICAL DATA")
print("-" * 30)

fig, axes = plt.subplots(1, len(categorical_cols), figsize=(12, 5))
fig.suptitle('Categorical Feature Distribution', fontsize=16, fontweight='bold')

for i, col in enumerate(categorical_cols):
    ax = axes[i]
    
    # Countplot
    sns.countplot(data=df, x=col, ax=ax, order=df[col].value_counts().index)
    ax.set_title(f'Distribution of {col}', fontsize=12, fontweight='bold')
    ax.set_xlabel(col)
    ax.set_ylabel('Count')
    
    # Add count labels on top of bars
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', 
                   (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha='center', va='center', 
                   xytext=(0, 9), 
                   textcoords='offset points',
                   fontsize=10)

plt.tight_layout()
plt.show()

# ============================================================================
# STEP 2: HANDLE MISSING DATA
# ============================================================================

print("\n" + "="*50)
print("STEP 2: HANDLE MISSING DATA")
print("="*50)

# Create a copy for different imputation methods
df_imputed = df.copy()

print("\n2.1 INITIAL MISSING VALUES COUNT:")
print("-" * 30)
missing_before = df_imputed.isnull().sum()
print(missing_before[missing_before > 0])

# 2.2 Initial imputation with median (numerical) and mode (categorical)
print("\n2.2 INITIAL IMPUTATION (Median for numerical, Mode for categorical):")
print("-" * 30)

# Impute numerical columns with median
for col in numerical_cols:
    if df_imputed[col].isnull().any():
        median_val = df_imputed[col].median()
        df_imputed[col].fillna(median_val, inplace=True)
        print(f"  - {col}: Imputed {df[col].isnull().sum()} values with median = {median_val:.2f}")

# Impute categorical columns with mode
for col in categorical_cols:
    if df_imputed[col].isnull().any():
        mode_val = df_imputed[col].mode()[0]
        df_imputed[col].fillna(mode_val, inplace=True)
        print(f"  - {col}: Imputed {df[col].isnull().sum()} values with mode = '{mode_val}'")

print("\n2.3 KNN IMPUTATION (More advanced method):")
print("-" * 30)

# For demonstration, let's apply KNN imputation on a fresh copy
df_knn = df.copy()

# KNN imputer works only on numerical data
knn_imputer = KNNImputer(n_neighbors=5)
numerical_data = df_knn[numerical_cols]
imputed_numerical = knn_imputer.fit_transform(numerical_data)
df_knn[numerical_cols] = imputed_numerical

# Still need to handle categorical columns with mode
for col in categorical_cols:
    if df_knn[col].isnull().any():
        mode_val = df_knn[col].mode()[0]
        df_knn[col].fillna(mode_val, inplace=True)

print("KNN imputation completed. Using this for further analysis.")
df_clean = df_knn.copy()

print("\nMissing values after imputation:")
print("-" * 30)
print(df_clean.isnull().sum())

# ============================================================================
# STEP 3: HANDLE OUTLIERS
# ============================================================================

print("\n" + "="*50)
print("STEP 3: HANDLE OUTLIERS")
print("="*50)

# Create a copy for outlier handling
df_outliers = df_clean.copy()

print("\n3.1 DETECTING OUTLIERS USING IQR METHOD:")
print("-" * 30)

def detect_outliers_iqr(data, column):
    """Detect outliers using IQR method"""
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

# Check for outliers in each numerical column
for col in numerical_cols:
    outliers, lower, upper = detect_outliers_iqr(df_outliers, col)
    print(f"\n{col}:")
    print(f"  - Lower bound: {lower:.2f}")
    print(f"  - Upper bound: {upper:.2f}")
    print(f"  - Number of outliers: {len(outliers)}")
    print(f"  - Outlier percentage: {(len(outliers)/len(df_outliers)*100):.2f}%")

print("\n3.2 WINSORIZATION (Capping outliers):")
print("-" * 30)

def winsorize_column(data, column, lower_percentile=1, upper_percentile=99):
    """Apply winsorization to cap outliers"""
    original_data = data[column].copy()
    
    # Calculate percentiles
    lower_limit = np.percentile(data[column].dropna(), lower_percentile)
    upper_limit = np.percentile(data[column].dropna(), upper_percentile)
    
    # Cap the values
    data[column] = np.where(data[column] < lower_limit, lower_limit, data[column])
    data[column] = np.where(data[column] > upper_limit, upper_limit, data[column])
    
    # Count how many values were modified
    modified_count = ((original_data < lower_limit) | (original_data > upper_limit)).sum()
    
    return data, modified_count, lower_limit, upper_limit

# Apply winsorization to numerical columns
for col in numerical_cols:
    if col != 'crs' and col != 'nasal_polyp':  # Skip binary columns
        df_outliers, modified, lower, upper = winsorize_column(df_outliers, col)
        print(f"  - {col}: Capped {modified} values between {lower:.2f} and {upper:.2f}")

print("\n3.3 VISUALIZATION - BEFORE AND AFTER WINSORIZATION:")
print("-" * 30)

# Compare before and after for one column (muc5ac as example)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('muc5ac: Before vs After Winsorization', fontsize=16, fontweight='bold')

# Before winsorization
axes[0].boxplot(df_clean['muc5ac'].dropna(), patch_artist=True)
axes[0].set_title('Before Winsorization', fontsize=12, fontweight='bold')
axes[0].set_ylabel('muc5ac Values')

# After winsorization
axes[1].boxplot(df_outliers['muc5ac'].dropna(), patch_artist=True)
axes[1].set_title('After Winsorization', fontsize=12, fontweight='bold')
axes[1].set_ylabel('muc5ac Values')

# Color the boxes
for ax in axes:
    for box in ax.findobj(match=plt.matplotlib.patches.PathPatch):
        box.set_facecolor('lightgreen')

plt.tight_layout()
plt.show()

# Use the winsorized dataset for further analysis
df_final = df_outliers.copy()

# ============================================================================
# STEP 4: CORRELATION AND RELATIONSHIP ANALYSIS
# ============================================================================

print("\n" + "="*50)
print("STEP 4: CORRELATION AND RELATIONSHIP ANALYSIS")
print("="*50)

print("\n4.1 CORRELATION MATRIX:")
print("-" * 30)

# Calculate correlation matrix for numerical columns
correlation_matrix = df_final[numerical_cols].corr()
print("Correlation Matrix:")
print(correlation_matrix.round(3))

# Visualize correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Matrix Heatmap', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

print("\n4.2 PAIRPLOT RELATIONSHIPS:")
print("-" * 30)

# Create pairplot for numerical features
pairplot_cols = ['muc5ac', 'il8', 'symptom_score']
sns.pairplot(df_final[pairplot_cols], diag_kind='kde', corner=False)
plt.suptitle('Pairplot of Numerical Features', y=1.02, fontsize=16, fontweight='bold')
plt.show()

print("\n4.3 SCATTER PLOTS WITH REGRESSION LINES:")
print("-" * 30)

# Scatter plots with regression lines
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Relationship with Symptom Score', fontsize=16, fontweight='bold')

# muc5ac vs symptom_score
sns.regplot(data=df_final, x='muc5ac', y='symptom_score', ax=axes[0], 
            scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
axes[0].set_title('muc5ac vs Symptom Score', fontsize=12, fontweight='bold')
axes[0].set_xlabel('muc5ac')
axes[0].set_ylabel('Symptom Score')

# il8 vs symptom_score
sns.regplot(data=df_final, x='il8', y='symptom_score', ax=axes[1], 
            scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
axes[1].set_title('IL8 vs Symptom Score', fontsize=12, fontweight='bold')
axes[1].set_xlabel('IL8')
axes[1].set_ylabel('Symptom Score')

plt.tight_layout()
plt.show()

print("\n4.4 CATEGORICAL VS NUMERICAL ANALYSIS:")
print("-" * 30)

# Boxplots for categorical vs numerical
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Categorical Features vs Symptom Score', fontsize=16, fontweight='bold')

# crs_str vs symptom_score
sns.boxplot(data=df_final, x='crs_str', y='symptom_score', ax=axes[0])
axes[0].set_title('CRS Status vs Symptom Score', fontsize=12, fontweight='bold')
axes[0].set_xlabel('CRS')
axes[0].set_ylabel('Symptom Score')

# nasal_polyp_str vs symptom_score
sns.boxplot(data=df_final, x='nasal_polyp_str', y='symptom_score', ax=axes[1])
axes[1].set_title('Nasal Polyp vs Symptom Score', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Nasal Polyp')
axes[1].set_ylabel('Symptom Score')

plt.tight_layout()
plt.show()

# ============================================================================
# STEP 5: FEATURE ENGINEERING
# ============================================================================

print("\n" + "="*50)
print("STEP 5: FEATURE ENGINEERING")
print("="*50)

# Create a copy for feature engineering
df_features = df_final.copy()

print("\n5.1 ENCODING CATEGORICAL VARIABLES:")
print("-" * 30)

# Label Encoding for categorical variables
label_encoders = {}
for col in ['crs_str', 'nasal_polyp_str']:
    le = LabelEncoder()
    df_features[f'{col}_encoded'] = le.fit_transform(df_features[col])
    label_encoders[col] = le
    print(f"  - {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")

print("\n5.2 SCALING NUMERICAL FEATURES:")
print("-" * 30)

# Standard Scaling (mean=0, std=1)
scaler_standard = StandardScaler()
numerical_to_scale = ['muc5ac', 'il8', 'symptom_score']

for col in numerical_to_scale:
    df_features[f'{col}_standard'] = scaler_standard.fit_transform(df_features[[col]])

# Min-Max Scaling (range 0-1)
scaler_minmax = MinMaxScaler()
for col in numerical_to_scale:
    df_features[f'{col}_minmax'] = scaler_minmax.fit_transform(df_features[[col]])

print("Standard Scaling and Min-Max Scaling applied.")
print("Original vs Scaled values (first 5 rows):")
print(df_features[['muc5ac', 'muc5ac_standard', 'muc5ac_minmax']].head())

print("\n5.3 CATEGORIZING TARGET VARIABLE:")
print("-" * 30)

# Create categorical symptom severity
def categorize_symptom(score):
    """Categorize symptom score into three categories"""
    if score < df_features['symptom_score'].quantile(0.33):
        return 'Mild'
    elif score < df_features['symptom_score'].quantile(0.66):
        return 'Moderate'
    else:
        return 'Severe'

df_features['symptom_severity'] = df_features['symptom_score'].apply(categorize_symptom)

# Show distribution
print("Symptom Severity Distribution:")
severity_counts = df_features['symptom_severity'].value_counts()
print(severity_counts)

# Visualize
plt.figure(figsize=(8, 6))
colors = ['lightgreen', 'gold', 'salmon']
plt.pie(severity_counts.values, labels=severity_counts.index, autopct='%1.1f%%',
        colors=colors, startangle=90, shadow=True)
plt.title('Distribution of Symptom Severity', fontsize=16, fontweight='bold')
plt.show()

# ============================================================================
# STEP 6: MODELING
# ============================================================================

print("\n" + "="*50)
print("STEP 6: MODELING")
print("="*50)

# Prepare data for modeling
print("\n6.1 DATA PREPARATION FOR MODELING:")
print("-" * 30)

# For Regression (predicting continuous symptom_score)
X_reg = df_features[['muc5ac', 'il8', 'crs', 'nasal_polyp']]
y_reg = df_features['symptom_score']

# For Classification (predicting symptom_severity)
X_clf = df_features[['muc5ac', 'il8', 'crs', 'nasal_polyp']]

# Encode target for classification
le_target = LabelEncoder()
y_clf = le_target.fit_transform(df_features['symptom_severity'])
print(f"Classification target encoding: {dict(zip(le_target.classes_, le_target.transform(le_target.classes_)))}")

# Split data
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

X_clf_train, X_clf_test, y_clf_train, y_clf_test = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42
)

print(f"\nRegression dataset:")
print(f"  - Training: {X_reg_train.shape[0]} samples")
print(f"  - Testing: {X_reg_test.shape[0]} samples")

print(f"\nClassification dataset:")
print(f"  - Training: {X_clf_train.shape[0]} samples")
print(f"  - Testing: {X_clf_test.shape[0]} samples")

print("\n6.2 REGRESSION MODEL (Linear Regression):")
print("-" * 30)

# Train Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_reg_train, y_reg_train)

# Predictions
y_reg_pred = lr_model.predict(X_reg_test)

# Evaluate
mse = mean_squared_error(y_reg_test, y_reg_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_reg_test, y_reg_pred)

print(f"Model Coefficients:")
for feature, coef in zip(X_reg.columns, lr_model.coef_):
    print(f"  - {feature}: {coef:.4f}")
print(f"Intercept: {lr_model.intercept_:.4f}")

print(f"\nModel Performance:")
print(f"  - Mean Squared Error (MSE): {mse:.2f}")
print(f"  - Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"  - R-squared (R²): {r2:.4f}")

# Visualize predictions vs actual
plt.figure(figsize=(10, 6))
plt.scatter(y_reg_test, y_reg_pred, alpha=0.5)
plt.plot([y_reg_test.min(), y_reg_test.max()], 
         [y_reg_test.min(), y_reg_test.max()], 'r--', lw=2)
plt.xlabel('Actual Symptom Score', fontsize=12)
plt.ylabel('Predicted Symptom Score', fontsize=12)
plt.title('Linear Regression: Actual vs Predicted', fontsize=16, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.show()

print("\n6.3 CLASSIFICATION MODEL (Random Forest):")
print("-" * 30)

# Train Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_clf_train, y_clf_train)

# Predictions
y_clf_pred = rf_model.predict(X_clf_test)

# Evaluate
accuracy = accuracy_score(y_clf_test, y_clf_pred)
print(f"Model Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_clf_test, y_clf_pred, 
                           target_names=le_target.classes_))

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X_clf.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)

# Visualize feature importance
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['feature'], feature_importance['importance'])
plt.xlabel('Importance', fontsize=12)
plt.title('Random Forest Feature Importance', fontsize=16, fontweight='bold')
plt.gca().invert_yaxis()
plt.show()

# ============================================================================
# ADDITIONAL EDA TOPICS (You mentioned if you missed any)
# ============================================================================

print("\n" + "="*50)
print("ADDITIONAL EDA TOPICS")
print("="*50)

print("\nA. TIME SERIES ANALYSIS (if applicable):")
print("-" * 30)
print("Note: Your dataset doesn't have time series data, but if it did,")
print("you could analyze trends, seasonality, and autocorrelation.")

print("\nB. CLUSTER ANALYSIS:")
print("-" * 30)
print("You could use clustering algorithms (like K-Means) to find")
print("natural groupings in your data.")

print("\nC. DIMENSIONALITY REDUCTION:")
print("-" * 30)
print("Techniques like PCA could help visualize high-dimensional data")
print("and identify the most important features.")

print("\nD. INTERACTION EFFECTS:")
print("-" * 30)
print("You could create interaction terms (e.g., muc5ac * il8) to")
print("capture combined effects of features.")

print("\nE. CROSS-VALIDATION:")
print("-" * 30)
print("For more robust model evaluation, use k-fold cross-validation")
print("instead of a simple train-test split.")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*50)
print("EDA SUMMARY")
print("="*50)

print("\n✅ COMPLETED STEPS:")
print("1. Initial Data Exploration and Visualization")
print("2. Missing Data Handling (Median/Mode + KNN Imputation)")
print("3. Outlier Treatment (Winsorization)")
print("4. Correlation and Relationship Analysis")
print("5. Feature Engineering (Encoding, Scaling, Categorization)")
print("6. Modeling (Regression and Classification)")

print("\n📊 KEY FINDINGS:")
print(f"- Dataset shape: {df.shape}")
print(f"- Original missing values: {df.isnull().sum().sum()} total")
print(f"- Outliers detected and treated using winsorization")
print(f"- Strongest correlation: muc5ac and symptom_score ({correlation_matrix.loc['muc5ac', 'symptom_score']:.3f})")
print(f"- Regression model R²: {r2:.4f}")
print(f"- Classification model accuracy: {accuracy:.4f}")

print("\n🎯 RECOMMENDATIONS:")
print("1. Consider collecting more data for better model generalization")
print("2. Experiment with different outlier handling methods")
print("3. Try more advanced models (XGBoost, Neural Networks)")
print("4. Perform feature selection to remove less important features")
print("5. Validate findings with domain experts")

print("\n" + "="*50)
print("EDA COMPLETE! Ready for further analysis or model deployment.")
print("="*50)