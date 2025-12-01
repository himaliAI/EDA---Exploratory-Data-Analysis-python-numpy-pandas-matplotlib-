# Correlation and Collinearity
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)

corr_matrix = df.corr()

# heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, cmap='coolwarm', center=0)
plt.title("Correlation Heatmap of Breast Cancer Features")
#plt.show()

# Variance inflation factor (VIF)
    # it quantifies multicollinearity among predictors
    # High VIF (>5 or 10) -> predictor is highly collinear with others
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
# add constant for regression
X = add_constant(df)
# VIF for each feature
vif_data = pd.DataFrame()
vif_data['Features'] = X.columns
vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(vif_data)