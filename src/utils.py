import pandas as pd
import numpy as np

def load_csv(path):
    """
    Load a csv file into a pandas DataFrame.
    args: path (str): path to the csv file
    returns: data (pd.DataFrame): the loaded csv file
    """
    data = pd.read_csv(path)
    return data

def split_dataset(df):
    """
    Split the data into features and target.
    args: df (pd.DataFrame): the input DataFrame
    returns: X (pd.DataFrame): the features
             y (pd.Series): the target
    """
    df = df.iloc[:, 1:]
    X = df.drop(columns=['Genotypes'])
    y = df['Genotypes']
    return X, y

def standardize(df):
    """
    Standardize the data in a DataFrame.
    args: df (pd.DataFrame): the input DataFrame
    returns: standardized (pd.DataFrame): the standardized DataFrame
    """
    standardized = (df - df.mean()) / df.std()
    return standardized

def count_columns(df):
    """
    Count the number of columns in a DataFrame.
    args: df (pd.DataFrame): the input DataFrame
    returns: n_columns (int): the number of columns
    """
    n_columns = df.shape[1]
    return n_columns

# mean columns in dataframe
def mean_columns(df):
    """
    Calculate the mean of each column in a DataFrame.
    args: df (pd.DataFrame): the input DataFrame
    returns: means (pd.Series): the mean of each column
    """
    means = df.mean()
    return means

def covariance_matrix(df):
    """
    Calculate the covariance matrix of a DataFrame.
    args: df (pd.DataFrame): the input DataFrame
    returns: cov (pd.DataFrame): the covariance matrix
    """
    mean = mean_columns(df)
    df = df - mean
    cov = df.T @ df / (len(df) - 1)
    return cov

def eigen_decomposition(df):
    """
    Perform eigen decomposition on the covariance matrix of a DataFrame.
    args: df (pd.DataFrame): the input DataFrame
    returns: eig_vals (np.array): the eigenvalues of the covariance matrix
             eig_vecs (np.array): the eigenvectors of the covariance matrix
    """
    cov = covariance_matrix(df)
    eig_vals, eig_vecs = np.linalg.eig(cov)
    return eig_vals, eig_vecs

def sort_eigenvalues(eigenvalues, eigenvectors):
    """
    Sort the eigenvalues and eigenvectors in descending order.
    args: eigenvalues (np.array): the eigenvalues to sort
          eigenvectors (np.array): the eigenvectors to sort
    returns: sorted_eigenvalues (np.array): the sorted eigenvalues
             sorted_eigenvectors (np.array): the sorted eigenvectors
    """
    # Sort the eigenvalues and eigenvectors in descending order
    idx = eigenvalues.argsort()[::-1]
    sorted_eigenvalues = eigenvalues[idx]
    sorted_eigenvectors = eigenvectors[:, idx]
    return sorted_eigenvalues, sorted_eigenvectors

def fit_transform(df, n_components):
    """
    Fit the PCA model and apply the dimensionality reduction to the DataFrame.
    args: df (pd.DataFrame): the input DataFrame
          n_components (int): the number of principal components to keep
    returns: transformed (pd.DataFrame): the transformed DataFrame
    """
    values, vectors = eigen_decomposition(df)
    sorted_values, sorted_vectors = sort_eigenvalues(values, vectors)
    W = sorted_vectors[:, :n_components]
    transformed = df @ W
    return transformed

def cumulative_variance_ratio(df, n_components):
    """
    Calculate the cumulative variance ratio of the PCA model.
    args: df (pd.DataFrame): the input DataFrame
          n_components (int): the number of principal components to keep
    returns: cumulative_variance (float): the cumulative variance ratio
    """
    values, vectors = eigen_decomposition(df)
    sorted_values, sorted_vectors = sort_eigenvalues(values, vectors)
    cumulative_variance = sum(sorted_values[:n_components]) / sum(sorted_values)
    return cumulative_variance

def explained_variance_ratio(df, n_components):
    """
    Calculate the explained variance ratio of the PCA model.
    args: df (pd.DataFrame): the input DataFrame
          n_components (int): the number of principal components to keep
    returns: explained_variance (np.array): the explained variance ratio
    """
    values, vectors = eigen_decomposition(df)
    sorted_values, sorted_vectors = sort_eigenvalues(values, vectors)
    explained_variance = sorted_values[:n_components] / sum(sorted_values)
    return explained_variance

# choose n_components base on threshold
def choose_n_components(df, threshold):
    """
    Choose the number of principal components based on a threshold.
    args: df (pd.DataFrame): the input DataFrame
          threshold (float): the threshold for the cumulative variance ratio
    returns: n_components (int): the number of principal components to keep
    """
    n_components = 1
    while cumulative_variance_ratio(df, n_components) < threshold:
        n_components += 1
    return n_components