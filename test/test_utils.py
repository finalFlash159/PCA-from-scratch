import sys
import os

# Thêm thư mục gốc vào sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils import load_csv, mean_columns, covariance_matrix, split_dataset, eigen_decomposition

# Import load_csv function from src.utils
file_path = 'data/wine.csv'
data = load_csv(file_path)
print(data.head())

X, y = split_dataset(data)
print(X.head())
print(y.head())


# cov_matrix = covariance_matrix(data)
# print(cov_matrix)
values, vectors = eigen_decomposition(X)
print(values)
print(vectors)
