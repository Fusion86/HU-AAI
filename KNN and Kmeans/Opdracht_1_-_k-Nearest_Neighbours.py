import numpy as np


class kNNAlgorithm:
    def __init__(
        self, trainingLabels, trainingData, validationLabels, validationData, extraData
    ):
        self.trainingLabels = trainingLabels
        self.trainingData = trainingData
        self.validationLabels = validationLabels
        self.validationData = validationData
        self.extraData = extraData
        self.k = 0
        self.passingProbability = 0

    def normaliseData(self, data):
        # calculate a min and a max for every parameter
        minimums = []
        maximums = []
        # for every different parameter
        for i in range(0, len(data[0])):
            currentMin = data[0][i]
            currentMax = data[0][i]
            # for every value of that parameter
            for j in range(1, len(data)):
                if data[j][i] < currentMin:
                    currentMin = data[j][i]
                elif data[j][i] > currentMax:
                    currentMax = data[j][i]
            minimums.append(currentMin)
            maximums.append(currentMax)

        # then normalise all data to that range
        for i in range(0, len(data)):
            for j in range(0, len(data[0])):
                data[i][j] = (data[i][j] - minimums[j]) / (maximums[j] - minimums[j])

    def calculateSquaredDistance(self, firstVector, secondVector):
        # return the distance squared between the vectors, they should be the same size
        count = 0
        for i in range(0, len(firstVector)):
            count += (firstVector[i] - secondVector[i]) * (
                firstVector[i] - secondVector[i]
            )

        return count

    # get a list of sorted distances and their labels and return a label with the given k
    def checkLabel(self, distances, k):
        labelCount = dict()
        labelCount["winter"] = 0
        labelCount["lente"] = 0
        labelCount["zomer"] = 0
        labelCount["herfst"] = 0

        for distance in range(0, k):
            labelCount[distances[distance][1]] += 1

        # get the keys with the maximum values
        keys = [
            key
            for m in [max(labelCount.values())]
            for key, val in labelCount.items()
            if val == m
        ]
        # tie breaker, remove the last counted value, so basically do a k - 1 check
        indexValueToRemove = k - 1
        while len(keys) > 1:
            labelCount[distances[indexValueToRemove][1]] -= 1
            keys = [
                key
                for m in [max(labelCount.values())]
                for key, val in labelCount.items()
                if val == m
            ]
            indexValueToRemove -= 1

        return keys[0]

    def trainAlgorithm(self):
        totalkValues = []
        for i in range(0, int(len(self.trainingData) / 4) - 1):
            totalkValues.append(0)

        # for every vector, calculate the distances to the other vector and get a value for every k for it
        for i in range(0, len(self.trainingData)):
            kValues = []
            # list of tuples with distances and label to the vector
            distances = []
            for j in range(0, len(self.trainingData)):
                if i != j:
                    distance = self.calculateSquaredDistance(
                        self.trainingData[i], self.trainingData[j]
                    )
                    distances.append((distance, self.trainingLabels[j]))

            # sort all distances on distance
            distances.sort(key=lambda tup: tup[0])

            # store for every k if the right label is assigned to this vector
            for k in range(1, int(len(self.trainingData) / 4)):
                kValues.append(self.checkLabel(distances, k) == self.trainingLabels[i])

            # add the new k values to the total k values
            for j in range(0, len(kValues)):
                totalkValues[j] += kValues[j]

        # and get the best k
        self.k = totalkValues.index(max(totalkValues)) + 1

    # use the best k on the validation data
    def validateAlgorithm(self):
        # the count that stores how many times the k got the validation data right
        count = 0
        for i in range(0, len(self.validationData)):
            distances = []
            for j in range(0, len(self.trainingData)):
                distance = self.calculateSquaredDistance(
                    self.validationData[i], self.trainingData[j]
                )
                distances.append((distance, self.trainingLabels[j]))

            distances.sort(key=lambda tup: tup[0])
            count += self.checkLabel(distances, self.k) == self.validationLabels[i]

        self.passingProbability = count / len(self.validationData)

    def checkExtraData(self):
        labels = []
        for i in range(0, len(self.extraData)):
            distances = []
            for j in range(0, len(self.trainingData)):
                distance = self.calculateSquaredDistance(
                    self.extraData[i], self.trainingData[j]
                )
                distances.append((distance, self.trainingLabels[j]))

            distances.sort(key=lambda tup: tup[0])
            labels.append(self.checkLabel(distances, self.k))

        print("the labels on the days.csv data were:")
        for label in labels:
            print(label)

    def run(self):
        self.normaliseData(self.trainingData)
        self.normaliseData(self.validationData)

        self.trainAlgorithm()

        self.validateAlgorithm()

        print("the best k = " + str(kNearest.k))
        print(
            "the probability on the validation data to correctly predict the season is = "
            + str(kNearest.passingProbability)
        )

        self.checkExtraData()


# get the training data
trainingData = np.genfromtxt(
    "KNN and Kmeans/dataset1.csv",
    delimiter=";",
    usecols=[1, 2, 3, 4, 5, 6, 7],
    converters={
        5: lambda s: 0 if s == b"-1" else float(s),
        7: lambda s: 0 if s == b"-1" else float(s),
    },
)

dates = np.genfromtxt("KNN and Kmeans/dataset1.csv", delimiter=";", usecols=[0])
trainingsLabels = []
for label in dates:
    if label < 20000301:
        trainingsLabels.append("winter")
    elif 20000301 <= label < 20000601:
        trainingsLabels.append("lente")
    elif 20000601 <= label < 20000901:
        trainingsLabels.append("zomer")
    elif 20000901 <= label < 20001201:
        trainingsLabels.append("herfst")
    else:  # from 01-12 to end of year
        trainingsLabels.append("winter")

# get the validation data
validationData = np.genfromtxt(
    "KNN and Kmeans/validation1.csv",
    delimiter=";",
    usecols=[1, 2, 3, 4, 5, 6, 7],
    converters={
        5: lambda s: 0 if s == b"-1" else float(s),
        7: lambda s: 0 if s == b"-1" else float(s),
    },
)

dates = np.genfromtxt("KNN and Kmeans/validation1.csv", delimiter=";", usecols=[0])
validatonLabels = []
for label in dates:
    if label < 20010301:
        validatonLabels.append("winter")
    elif 20010301 <= label < 20010601:
        validatonLabels.append("lente")
    elif 20010601 <= label < 20010901:
        validatonLabels.append("zomer")
    elif 20010901 <= label < 20011201:
        validatonLabels.append("herfst")
    else:  # from 01-12 to end of year
        validatonLabels.append("winter")

# get the extra data
extraData = np.genfromtxt(
    "KNN and Kmeans/days.csv",
    delimiter=";",
    usecols=[1, 2, 3, 4, 5, 6, 7],
    converters={
        4: lambda s: 0 if s == b"-1" else float(s),
        6: lambda s: 0 if s == b"-1" else float(s),
    },
)

kNearest = kNNAlgorithm(
    trainingsLabels, trainingData, validatonLabels, validationData, extraData
)
kNearest.run()
