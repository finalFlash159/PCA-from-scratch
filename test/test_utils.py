import sys
import os

# Thêm thư mục gốc vào sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils import load_csv, mean_columns, covariance_matrix

# Import load_csv function from src.utils
file_path = 'data/wine.csv'
data = load_csv(file_path)
print(data.head())


# cov_matrix = covariance_matrix(data)
# print(cov_matrix)

