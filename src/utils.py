import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

def load_csv(path):
    """
    Load a csv file into a pandas DataFrame.
    args: path (str): path to the csv file
    returns: data (pd.DataFrame): the loaded csv file
    """
    data = pd.read_csv(path)
    return data

def split_dataset(df, target_column):
    """
    Split the data into features and target.
    args: df (pd.DataFrame): the input DataFrame
    returns: X (pd.DataFrame): the features
             y (pd.Series): the target
    """
    df = df.iloc[:, 1:]
    X = df.drop(columns=target_column)
    y = df[target_column]
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

def label_encoder(df, target_column):
    """
    Encode the target column of a DataFrame.
    args: df (pd.DataFrame): the input DataFrame
          target_column (str): the column to encode
    returns: df (pd.DataFrame): the DataFrame with the encoded target column
    """
    df[target_column] = df[target_column].astype('category')
    df[target_column] = df[target_column].cat.codes
    return df

def scatter_plot(df, target_column):
    """
    Create a scatter plot of the original data.
    args: df (pd.DataFrame): the input DataFrame
          target_column (str): the column to use as the target
    """
    sns.pairplot(df, hue=target_column)
    plt.show()

def scatter_plot_2d_px(df, target_column):
    """
    Create a scatter plot of the original data using Plotly.
    args: df (pd.DataFrame): the input DataFrame
          target_column (str): the column to use as the target
    """
    fig = px.scatter(
        df,
        x=df.columns[0],
        y=df.columns[1],
        color=target_column,
        title="Scatter Plot",
        labels={df.columns[0]: df.columns[0], df.columns[1]: df.columns[1]}
    )
    fig.show()

def scatter_plot_3d(df, target_column):
    """
    Create a 3D scatter plot of the original data.
    args: df (pd.DataFrame): the input DataFrame
          target_column (str): the column to use as the target
    """
    df = label_encoder(df, target_column)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2], c=df[target_column])
    plt.show()



def scatter_plot_3d_px(df, target_column):
    """
    Create a 3D scatter plot of the original data.
    args: 
        df (pd.DataFrame): the input DataFrame
        target_column (str): the column to use as the target
    """
    # Lấy tên cột cho các trục
    x_column = df.columns[0]
    y_column = df.columns[1]
    z_column = df.columns[2]

    # Tạo biểu đồ 3D scatter
    fig = px.scatter_3d(
        df,
        x=x_column, 
        y=y_column, 
        z=z_column, 
        color=target_column,  # Dùng target_column để phân biệt màu sắc
        title="3D Scatter Plot",
        labels={x_column: x_column, y_column: y_column, z_column: z_column}
    )
    fig.show()

def explained_variance_plot(df, n_components):
    """
    Create a bar plot of the explained variance ratio.
    args: df (pd.DataFrame): the input DataFrame
          n_components (int): the number of principal components
    """
    explained_variance = explained_variance_ratio(df, n_components)
    plt.bar(range(1, n_components + 1), explained_variance)
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Explained Variance Ratio')
    plt.show()

def explained_variance_px(df, n_components):
    """
    Create a bar plot of the explained variance ratio using Plotly.
    args: df (pd.DataFrame): the input DataFrame
          n_components (int): the number of principal components
    """
    explained_variance = explained_variance_ratio(df, n_components)
    fig = px.area(
        x=range(1, n_components + 1),
        y=explained_variance,
        labels={"x": "Principal Component", "y": "Explained Variance Ratio"},
        title="Explained Variance Ratio"
    )
    # Cập nhật trục x để chỉ hiển thị giá trị nguyên
    fig.update_xaxes(
        tickmode="array",  # Đặt chế độ hiển thị theo danh sách cụ thể
        tickvals=list(range(1, n_components + 1))  # Chỉ hiển thị các giá trị nguyên
    )
    fig.show()

def cumulative_variance_plot(df, n_components):
    """
    Create a line plot of the cumulative variance ratio.
    args: df (pd.DataFrame): the input DataFrame
          n_components (int): the number of principal components
    """
    cumulative_variance = [cumulative_variance_ratio(df, i) for i in range(1, n_components + 1)]
    plt.plot(range(1, n_components + 1), cumulative_variance)
    plt.xlabel('Principal Component')
    plt.ylabel('Cumulative Variance Ratio')
    plt.title('Cumulative Variance Ratio')
    plt.show()

def cumulative_variance_px(df, n_components):
    """
    Create a line plot of the cumulative variance ratio using Plotly.
    args: df (pd.DataFrame): the input DataFrame
          n_components (int): the number of principal components
    """
    cumulative_variance = [cumulative_variance_ratio(df, i) for i in range(1, n_components + 1)]
    fig = px.area(
        x=range(1, n_components + 1),
        y=cumulative_variance,
        labels={"x": "Principal Component", "y": "Cumulative Variance Ratio"},
        title="Cumulative Variance Ratio"
    )
    # Cập nhật trục x để chỉ hiển thị giá trị nguyên
    fig.update_xaxes(
        tickmode="array",  # Đặt chế độ hiển thị theo danh sách cụ thể
        tickvals=list(range(1, n_components + 1))  # Chỉ hiển thị các giá trị nguyên
    )
    fig.show()

#  exp_var_cumul = np.cumsum(pca.explained_variance_ratio_)

# px.area(
#     x=range(1, exp_var_cumul.shape[0] + 1),
#     y=exp_var_cumul,
#     labels={"x": "# Components", "y": "Explained Variance"}
# )
