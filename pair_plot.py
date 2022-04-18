from matplotlib import pyplot as plt
import pandas as pd

if __name__ == "__main__":
    df = pd.read_csv('datasets/dataset_train.csv')
    df = df.select_dtypes(include='number')
    df = df.dropna(axis=1, how='all')
    df.drop('Index', inplace=True, axis=1)

    #create scatter plot matrix for all the features
    pd.plotting.scatter_matrix(df, alpha=0.2, figsize=(12, 12), diagonal='kde')
    plt.show()