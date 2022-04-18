from matplotlib import pyplot as plt
import pandas as pd


if __name__ == "__main__":
	df = pd.read_csv('datasets/dataset_train.csv')
	df = df.select_dtypes(include='number')
	df = df.dropna(axis=1, how='all')
	df.drop('Index', inplace=True, axis=1)

	courses = df.columns.values
	df = pd.read_csv('datasets/dataset_train.csv')
	for i in range(len(courses)):
		df[courses[i]].hist(by=df['Hogwarts House'],legend=courses[i])
	plt.show()