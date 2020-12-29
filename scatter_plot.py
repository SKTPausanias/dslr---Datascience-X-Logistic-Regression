from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

if __name__ == "__main__":
	df = pd.read_csv('datasets/dataset_train.csv')
	df = df.select_dtypes(include='number')
	df = df.dropna(axis=1, how='all')
	#df.drop('Index', inplace=True, axis=1)

	colnames = list(df.columns)
	for i in range(1, len(colnames)):
		df.reset_index().plot(x="Index", y=colnames[i], kind = 'line', sharex = True, ls="none", marker="o", markersize=3, legend=colnames[i])
	plt.show()