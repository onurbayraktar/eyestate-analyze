import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import preprocessing

COLUMNS = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
            "O2", "P8", "T8", "FC6", "F4", "F8", "AF4", "STATE"]


def createConnectionWithDB(dbPath):
    conn = sqlite3.connect(dbPath)
    return conn


def closeConnectionWithDB(conn):
    conn.close()


def createDataFrame(connectionObject, query):
    df = pd.read_sql_query(query, connectionObject)
    fillNullValuesIfExists(df)
    return df


def fillNullValuesIfExists(df):
    if df.isnull().values.any():
        print("There exists null values..")
        for column in COLUMNS:
            df[column].fillna(df[column].mean(), inplace=True)


# Function to plotting graphs to see the distribution of the values. We'll use it to check outliers. #
def plotTheDataDistribution(df):
    values = df.values
    # create a subplot for each time series
    plt.figure(figsize=(10, 20))
    for i in range(values.shape[1]):
        columnTitle = COLUMNS[i]
        ax = plt.subplot(values.shape[1], 1, i + 1)
        ax.set_ylabel(columnTitle)
        plt.plot(values[:, i])
    plt.show()


# Function to handle the outliers seen in the data; we need to delete the outlier value; and fill it with
# mean value of corresponding field. #
def outlierHandling(df):
    # For each of column, we'll calculate the mean/std/median/limits #
    for i in range(df.shape[1]-1):
        rowCounter = 0
        columnTitle = COLUMNS[i]
        meanOfColumn = df[columnTitle].mean()
        sdOfColumn = df[columnTitle].std()
        median = df[columnTitle].median()
        distanceFromMean = sdOfColumn * 3
        lowerLimit = meanOfColumn - distanceFromMean
        upperLimit = meanOfColumn + distanceFromMean
        for value in df[columnTitle]:
            if value > upperLimit or value < lowerLimit:    # If the value is out of boundaries; i's an outlier
                df.iat[rowCounter,i] = median
            rowCounter += 1
    return df


def main():
    database = "EyeState.db"
    fetchQuery = "SELECT * FROM States"

    conn = createConnectionWithDB(database)
    dataFrame = createDataFrame(conn, fetchQuery)
    plotTheDataDistribution(dataFrame)              # The distribution before outlier handling.
    dataFrame = outlierHandling(dataFrame)
    plotTheDataDistribution(dataFrame)              # The distribution after outlier handling.


if __name__ == '__main__':
    main()
