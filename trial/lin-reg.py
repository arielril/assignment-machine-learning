# import cell
from sklearn import datasets
import matplotlib.pyplot as plt
# from tqdm import tqdm_notebook as tqdm
import time
import numpy as np
import pandas as pd

SEED = 200

def plot_regression_line(x, y):
    # plotting the actual points as scatter plot
    plt.scatter(x, y, color="m",
                marker="o", s=30)

    # plotting data
    # plt.plot(x, y, color="g")

    # putting labels
    plt.xlabel('x')
    plt.ylabel('y')

    # function to show plot
    plt.show()

def minmax(x, column):
    diff = max(column) - min(column)
    return (x - min(column)) / diff

def rescale(df, columns):
    # for each column
    for c in columns:
        # rescale each value (row)
        df[c] = df.apply(lambda row: minmax(row[c], df[c]), axis=1)

    return df

# load csv cell...
"""Nao alterar"""
df = pd.read_csv('../regression_dataset/train_reg.csv').astype(dtype='float64')
# print(df.get('x'))
df = rescale(df, columns=df.columns[:])
print(df)
# plot_regression_line(df.get('x'), df.get('y'))

#df = rescale(df, columns=df.columns[:])

# Try the normal equation for the first 200
#subset = df.iloc[0:200]
#X, Y = subset.iloc[:,0], subset.iloc[:,1]
