import sys
import pandas as pd
import numpy as np

def mean_(x):
	x = x[~np.isnan(x)]
	return (float(sum(x)) / len(x))

def count_(x):
	x = x[~np.isnan(x)]
	return len(x)

def std_(x):
    x = x[~np.isnan(x)]
    m = mean_(x)
    res = 0.0
    for i in x:
        res += ((i - m) * (i - m))
    res = np.sqrt(res / len(x))
    return res

def min_(x):
	x = x[~np.isnan(x)]
	min = x[0]
	for i in x:
		if i < min:
			min = i
	return min

def max_(x):
	x = x[~np.isnan(x)]
	max = x[0]
	for i in x:
		if i > max:
			max = i
	return max

def median_(x):
	x = x[~np.isnan(x)]
	n = len(x)
	x = x.sort_values(ignore_index=True)
	if n % 2 == 0:
		median1 = x[ n // 2]
		median2 = x[( n// 2) - 1]
		median = (median1 + median2) / 2.0
	else:
		median = float(x[n // 2])
	return median

def quartiles_(x, percentile):
	x = x[~np.isnan(x)]
	x = x.sort_values(ignore_index=True)
	if len(x) % 2 == 0:
		median1 = x[:(len(x)//2)]
		median2 = x[len(x)//2:]
		if percentile == 25:
			return median_(median1)
		elif percentile == 75:
			return median_(median2)
	else:
		median1 = x[:(len(x)//2)]
		median2 = x[len(x)//2 + 1:]
		if percentile == 25:
			return median_(median1)
		elif percentile == 75:
			return median_(median2)

if __name__ == "__main__":
	if len(sys.argv) != 2:
		print("usage: ./describe.py [dataset].csv")
		exit(1)
	try:
		df = pd.read_csv(sys.argv[1])
	except:
		print("Unable to open/find that file")
		exit(1)
	df = df.select_dtypes(include='number')
	df = df.dropna(axis=1, how='all')
	df.drop('Index', inplace=True, axis=1)

	str1 = "".ljust(10)
	for feature in df.columns.values:
		str1 += str('{:.10}'.format(feature)).rjust(10) + ' '
	print(str1)
	dfmean = df.apply(count_, axis=0)
	dfmean = dfmean.to_frame()
	str1 = "Count".ljust(10)
	for i in dfmean[0]:
		str1 += str(i).rjust(10) + ' '
	print(str1)

	dfmean = df.apply(mean_, axis=0)
	dfmean = dfmean.to_frame()
	str1 = "Mean".ljust(10)
	for i in dfmean[0]:
		str1 += str("{:.2f}".format(i)).rjust(10) + ' '
	print(str1)

	dfmean = df.apply(std_, axis=0)
	dfmean = dfmean.to_frame()
	str1 = "Std".ljust(10)
	for i in dfmean[0]:
		str1 += str("{:.2f}".format(i)).rjust(10) + ' '
	print(str1)

	dfmean = df.apply(min_, axis=0)
	dfmean = dfmean.to_frame()
	str1 = "Min".ljust(10)
	for i in dfmean[0]:
		str1 += str("{:.2f}".format(i)).rjust(10) + ' '
	print(str1)

	dfmean = df.apply(quartiles_, percentile=25, axis=0)
	dfmean = dfmean.to_frame()
	str1 = "25%".ljust(10)
	for i in dfmean[0]:
		str1 += str("{:.2f}".format(i)).rjust(10) + ' '
	print(str1)

	dfmean = df.apply(median_, axis=0)
	dfmean = dfmean.to_frame()
	str1 = "50%".ljust(10)
	for i in dfmean[0]:
		str1 += str("{:.2f}".format(i)).rjust(10) + ' '
	print(str1)

	dfmean = df.apply(quartiles_, percentile=75, axis=0)
	dfmean = dfmean.to_frame()
	str1 = "75%".ljust(10)
	for i in dfmean[0]:
		str1 += str("{:.2f}".format(i)).rjust(10) + ' '
	print(str1)

	dfmean = df.apply(max_, axis=0)
	dfmean = dfmean.to_frame()
	str1 = "Max".ljust(10)
	for i in dfmean[0]:
		str1 += str("{:.2f}".format(i)).rjust(10) + ' '
	print(str1)
	exit(0)