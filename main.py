import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn import tree, metrics
from sklearn.naive_bayes import GaussianNB


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


# Function that splits the data into train / test pairs. #
def generateTrainTestData(df):
    y_data = df['eyeDetection']
    x_data = df.drop(['eyeDetection'], axis=1)

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, shuffle=True)
    trainTestData = [x_train, x_test, y_train, y_test]
    return trainTestData


def applyDecisionTreeAlgorithm(trainTestData):
    x_train = trainTestData[0]
    y_train = trainTestData[2]
    x_test = trainTestData[1]
    y_test = trainTestData[3]

    classifier = tree.DecisionTreeClassifier(criterion="entropy")
    classifier = classifier.fit(x_train,y_train)
    prediction = classifier.predict(x_test)
    result = [y_test, prediction]
    return result


def applyNaiveBayesAlgorithm(trainTestData):
    x_train = trainTestData[0]
    y_train = trainTestData[2]
    x_test = trainTestData[1]
    y_test = trainTestData[3]

    classifier = GaussianNB()
    classifier.fit(x_train, y_train)
    prediction = classifier.predict(x_test)
    result = [y_test, prediction]
    return result


def evaluation(result):
    testedValues = result[0]
    predictedValues = result[1]
    accuracy = metrics.accuracy_score(testedValues, predictedValues)
    print("Accuracy: " + str(accuracy))


def main():
    database = "EyeState.db"
    fetchQuery = "SELECT * FROM States"

    conn = createConnectionWithDB(database)
    dataFrame = createDataFrame(conn, fetchQuery)
    #plotTheDataDistribution(dataFrame)              # The distribution before outlier handling.
    dataFrame = outlierHandling(dataFrame)
    #plotTheDataDistribution(dataFrame)              # The distribution after outlier handling.
    trainTestData = generateTrainTestData(dataFrame)
    results = applyDecisionTreeAlgorithm(trainTestData)
    #results = applyNaiveBayesAlgorithm(trainTestData)
    evaluation(results)



if __name__ == '__main__':
    main()
