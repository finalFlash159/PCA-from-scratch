import sys
import os

# Thêm thư mục gốc vào sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import load_csv, split_dataset, standardize, count_columns, fit_transform, cumulative_variance_ratio, explained_variance_ratio, choose_n_components

# Load dataset
file_path = 'data/wine.csv'
data = load_csv(file_path)
print("Column names: ", data.columns)

# Split dataset
X, y = split_dataset(data)
print(X.columns) 

# Standardize the data
X_standardized = standardize(X)
print(X_standardized.head())

# Count the number of columns
n_columns = count_columns(X)
print(f"Number of columns: {n_columns}") #17

# Calculate cumulative variance ratio
cum_variance_ratio = cumulative_variance_ratio(X_standardized, n_components=17)
print("Comulative variance ratio: ", cum_variance_ratio)

# Choose n_components based on threshold = 0.95
n_components = choose_n_components(X_standardized, threshold=0.95)
print("Number of components: ", n_components) # 14

# Calculate explained variance ratio
explained_variance_ratio = explained_variance_ratio(X_standardized, n_components)
print("Explained variance ratio: ", explained_variance_ratio)

# Calculate cụmulative variance ratio
cum_variance_ratio = cumulative_variance_ratio(X_standardized, n_components)
print("Cumulative variance ratio: ", cum_variance_ratio)

# Fit and transform the data
X_pca = fit_transform(X_standardized, n_components)
print(X_pca.head())

