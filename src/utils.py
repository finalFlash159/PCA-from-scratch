import pandas as pd

def load_csv(path):
    """
    Load a csv file into a pandas DataFrame.
    args: path (str): path to the csv file
    returns: data (pd.DataFrame): the loaded csv file
    """
    data = pd.read_csv(path)
    return data

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

