from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

if __name__ == "__main__":
	df = pd.read_csv('datasets/dataset_train.csv')
	df = df.select_dtypes(include='number')
	df = df.dropna(axis=1, how='all')
	df.drop('Index', inplace=True, axis=1)

	colnames = list(df.columns)
	for i in range(len(colnames)):
		for j in range(i+1,len(colnames)):
			# plot only Defense Against the Dark Arts and Astronomy
			#if colnames[j] == 'Defense Against the Dark Arts' and colnames[i] == 'Astronomy':
				plt.scatter(df[colnames[i]],df[colnames[j]])
				plt.xlabel(colnames[i])
				plt.ylabel(colnames[j])
				plt.show()

# Defense Against the Dark Arts and Astronomy are similar