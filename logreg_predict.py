import pandas as pd
import sys
import numpy as np
from my_logistic_regression import MyLogisticRegression as MyLR

if __name__ == "__main__":
    #take 2 parameter csv
    if len(sys.argv) != 3:
        print("Usage: python logreg_predict.py <data_file> <weight_file>")
        sys.exit(1)
    
    df = pd.read_csv(sys.argv[1])
    if (df is None):
        print("Error: could not read csv file")
        sys.exit(1)
    
    weights = pd.read_csv(sys.argv[2])
    if (weights is None):
        print("Error: could not read csv file")
        sys.exit(1)

    #manage data
    df.fillna(method='ffill',inplace=True, axis=0)
    df.drop('Index', inplace=True, axis=1)
    #conver best hand to numerical values
    df['Best Hand'] = df['Best Hand'].map({'Left': 0, 'Right': 1})
    df = df.select_dtypes(include='number')
    df.drop('Hogwarts House', inplace=True, axis=1)

    #normalize data for each column
    df = (df - df.mean()) / df.std()
    #convert X to numpy array
    X = np.array(df)

    #predict house of each student with weights
    houses = weights.columns.values
    y_hatdf = pd.DataFrame(columns=houses)
    for house in houses:
        #create logistic regression object
        thetas = np.array(weights[house])
        #reshape thetas to be a column vector
        thetas = thetas.reshape(len(thetas), 1)
        lr = MyLR(thetas)
        #predict house of each student
        y_hat = lr.predict_(X)
        y_hat = y_hat.reshape(len(y_hat))
        #add column to dataframe
        y_hatdf[house] = y_hat
    #select the house with the highest probability
    y_hatdf['predicted_house'] = y_hatdf.idxmax(axis=1)

    #save house predictions to csv
    house_df = y_hatdf.loc[:, ['predicted_house']]
    house_df.columns = ['Hogwarts House']
    house_df.index.name = 'Index'
    house_df.to_csv('houses.csv', index=True)



    