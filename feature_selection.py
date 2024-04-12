import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold,SelectKBest, chi2
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler

## At first, we would like to use the correlation matrix to select the features
df1 = pd.read_csv('data/cleaned.csv')
correlation_matrix = df1.corr()

# Isolate the 'popularity' correlations and drop the self-correlation
popularity_correlation = correlation_matrix['popularity'].drop('popularity', axis=0)

# Sort the correlations in descending order
sorted_correlation = popularity_correlation.sort_values(ascending=False)

# Print the features with a correlation higher than 0.01
for feature, correlation in sorted_correlation.items():
    if abs(correlation) >= 0.01:
        print(feature)



## Secondly, I would like to use the VarianceThreshold method to select the features
df_feature = df1.drop(columns=['popularity'])
threshold_values = np.linspace(0, 1, 11)  

# Prepare lists to store the results
thresholds = []
num_low_variance_cols = []
low_variance_cols_list = []  

for i in threshold_values:
    var_thres = VarianceThreshold(threshold=i)  
    var_thres.fit(df_feature)

    # Identify columns that do not meet the current variance threshold
    low_variance_cols = [column for column in df_feature.columns
                         if column not in df_feature.columns[var_thres.get_support()]]

    # Append the results to the lists to perform analysis
    thresholds.append(i)
    num_low_variance_cols.append(len(low_variance_cols))
    low_variance_cols_list.append(low_variance_cols)  

# Plot results to visualize the number of low variance columns at each threshold
plt.figure(figsize=(10, 6))
plt.plot(thresholds, num_low_variance_cols, marker='o', linestyle='-')
plt.title('Number of Low Variance Columns vs. Variance Threshold')
plt.xlabel('Variance Threshold')
plt.ylabel('Number of Low Variance Columns')
plt.yticks(range(0, max(num_low_variance_cols) + 1, max(num_low_variance_cols) // 10 + 1))
plt.grid(True)
plt.show()

# Display the low variance columns for each threshold
for t, names in zip(thresholds, low_variance_cols_list):
    print(f"Variance Threshold: {t:.1f} -> Low Variance Columns: {names}")

# According to the plot, we would like to choose the features with threshold as 0.4 - 0.7 as the chosen features
vt_features = low_variance_cols_list[4]
print("Variance Threshold selected features: ",vt_features)


## Thirdly, I would like to use the PCA method to select the features
data = df1.copy()

# We found that the 'bathrooms_text' column contains text data which is duplicated by the 'bathrooms' column
data.drop(columns=['bathrooms_text'], inplace=True)

# Separate features and target
X = data.drop('popularity', axis=1)
y = data['popularity']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA to determine the optimal number of components
pca = PCA()
X_pca = pca.fit_transform(X_scaled)
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_explained_variance = np.cumsum(explained_variance_ratio)

# Find the number of components for 95% variance
optimal_components = np.where(cumulative_explained_variance >= 0.95)[0][0] + 1

# Plot the explained variance to visualize the trade-off
plt.figure(figsize=(8, 5))
plt.plot(cumulative_explained_variance, label='Cumulative Explained Variance')
plt.axhline(y=0.95, color='red', linestyle='--', linewidth=2, label='95% explained variance')
plt.axvline(x=optimal_components - 1, color='red', linestyle='--', linewidth=2, label=f'Optimal components: {optimal_components}')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance by Components')
plt.legend()
plt.grid(True)
plt.show()

# Apply PCA with the optimal number of components
pca_optimal = PCA(n_components=optimal_components)
X_pca_optimal = pca_optimal.fit_transform(X_scaled)

# Get the loadings of the PCA components
loadings = pca_optimal.components_
loadings_df = pd.DataFrame(loadings.T, columns=[f'PC{i+1}' for i in range(loadings.shape[0])], index=X.columns)

# Calculate the absolute sum of loadings for each feature to see their overall contribution
feature_importance = np.abs(loadings_df).sum(axis=1).sort_values(ascending=False)
pca_features = feature_importance[0:23].index
# Display the feature importance according to the PCA plot
print("PCA selected features: ",pca_features)



## Finally, We would like to use the Chi-Square test to select the features
# Convert features to be non-negative if not already
scaler_mm = MinMaxScaler()
X_scaled_non_negative = scaler_mm.fit_transform(X_scaled)

## Based on previous analysis, we would like to set k as mean of the number of features selected by PCA and VarianceThreshold
k = 19  
chi_selector = SelectKBest(chi2, k=k)
X_kbest_features = chi_selector.fit_transform(X_scaled_non_negative, y)

# Get the selected feature indices and names
selected_indices = chi_selector.get_support(indices=True)
chi_features = X.columns[selected_indices]

print("Chi-square selected features:", chi_features)


## In conclusion, we would like to choose the features selected by PCA, VarianceThreshold and Chi-Square test
## These features are the most important features which selected by three different methods
print("Common features between all three methods \n",set(vt_features)&set(pca_features)&set(chi_features))