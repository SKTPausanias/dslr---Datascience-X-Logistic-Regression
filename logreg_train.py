import pandas as pd
import sys
import numpy as np
from my_logistic_regression import MyLogisticRegression as MyLR
from sklearn.metrics import accuracy_score


if __name__ == "__main__":
    #take csv as parameter for this script
    if len(sys.argv) < 2:
        print("Usage: python logreg_train.py <csv_file>")
        sys.exit(1)
    
    #read csv file
    df = pd.read_csv(sys.argv[1])
    if (df is None):
        print("Error: could not read csv file")
        sys.exit(1)

    #manage data
    df.fillna(method='ffill',inplace=True, axis=0)
    #df.dropna(axis=1, how='all', inplace=True)
    df.drop('Index', inplace=True, axis=1)
    #conver best hand to numerical values
    df['Best Hand'] = df['Best Hand'].map({'Left': 0, 'Right': 1})
    
    
    #perform logistic regression fit for each 'Hogwarts House' and save weights into output file
    #create csv file to save weights from houses
    df_weights = pd.DataFrame()

    houses = df['Hogwarts House'].unique()
    for house in houses:
        # y array of 1 and 0, 1 if the student belongs to the house, 0 otherwise, y as numpy array
        y = np.array(df['Hogwarts House'] == house).astype(int)
        #reshape y to be a column vector
        y = y.reshape(len(y), 1)
        #drop non-numeric columns
        X = df.select_dtypes(include='number')
        #normalize data for each column
        X = (X - X.mean()) / X.std()
        #convert X to numpy array
        X = np.array(X)

        #create logistic regression object
        thetas = np.array([1] * (X.shape[1] + 1)).reshape((X.shape[1] + 1), 1)
        lr = MyLR(thetas, 0.5, 10000)
        #perform gradient descent
        lr.fit_(X, y)
        #creating dataframe wih weights for each house
        df_house = pd.DataFrame(data=lr.thetas, columns=[house])
        df_weights = pd.concat([df_weights, df_house], axis=1)
    
    #save weights to csv file
    #print(df_weights)
    f = open('./logreg_weights.csv', 'w')
    df_weights.to_csv(f, index=False)
    f.close()

    #predict for each house
    #print X start
    houses = df_weights.columns.values
    y_hatdf = pd.DataFrame(columns=houses)
    print(houses)
    for house in houses:
        #create logistic regression object
        thetas = np.array(df_weights[house])
        #reshape thetas to be a column vector
        thetas = thetas.reshape(len(thetas), 1)
        lr = MyLR(thetas)
        #predict house of each student
        y_hat = lr.predict_(X)
        y_hat = y_hat.reshape(len(y_hat))
        #add column to dataframe
        y_hatdf[house] = y_hat
    y_hatdf['predicted_house'] = y_hatdf.idxmax(axis=1)

    accuracy = accuracy_score(df['Hogwarts House'], y_hatdf['predicted_house'])
    print("Accuracy: %.3f" % accuracy)

