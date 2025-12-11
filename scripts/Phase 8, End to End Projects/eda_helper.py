# Helper function for Reusable EDA
import pandas as pd

# missing values summary
def missing_summary(df):
    # returns a DataFrame with count and % of missing values per column
    missing_count = df.isnull().sum()
    missing_percent = (missing_count / len(df)) * 100
    summary = pd.DataFrame({
        'Missing Count': missing_count,
        'Missing Percent': missing_percent.round(2)
    })
    return summary[summary['Missing Count'] > 0].sort_values(by='Missing Percent', ascending=False)

# Outlier summary
def outlier_summary(df, method='IQR'):
    # returns outlier counts per numeric column using IQR (default) or Z-score method
    outliers = {}
    numeric_cols = df.select_dtypes(include="number").columns
        # or numeric_cols = df.select_dtypes(include=['int', 'float']).columns
    
    for col in numeric_cols:
        series = df[col].dropna()
        if method == 'IQR':
            Q1, Q3 = series.quantile([0.25, 0.75])
            IQR = Q3 - Q1 
            lower, upper = Q1 - (1.5 * IQR), Q3 + (1.5 * IQR)
            outliers[col] = ((series < lower) | (series > upper)).sum()
        elif method == 'Z':
            mean, std = series.mean(), series.std()
            outliers[col] = (abs(series - mean) > 3 * std).sum()

    return pd.DataFrame.from_dict(outliers, orient='index', columns=['Outlier Count']).sort_values(by="Outlier Count", ascending=False)
    
# Category report
def category_report(df):
    # returns frequency table for caegorical columns
    cat_cols = df.select_dtypes(include="object").columns
    report = {}
    for col in cat_cols:
        report[col] = df[col].value_counts().to_frame().rename(columns={col:'Count'})
    return report 
    # report is a dictionary of frequency tables for each categorical variable

# Mini EDA Report Builder
def eda_report(df): 
    # prints a quick EDA report: shape, missing summary, outlier summary, and categorical report
    print(f"Dataset Shape: {df.shape}")
    print(f"\nMissing Values Summary:\n{missing_summary(df)}")
    print(f"\nOutliner Summary:\n{outlier_summary(df)}")
    print(f"\nCategorical Report:")
    cat_report = category_report(df)
    for col, table in cat_report.items():
        print(f"\nColumn: {col}")
        print(table.head(10))