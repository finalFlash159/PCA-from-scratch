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
    X = df.drop(columns=['Genotypes'])
    y = df['Genotypes']
    return X, y

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
