import time, os, sys
from river import tree, metrics, preprocessing
import matplotlib.pyplot as plt
import pandas 
import numpy as np
import math

class financialAI():
    def __init__(self, dataFile="main/data.csv"):
        """This function contains the features we learn from and the ML classifier we are using """
        self.file = dataFile
        self.features = ["SP500", "Consumer Price Index", "Long Interest Rate"] 

        # arrays for graphs
        self.incorrectPredictions, self.correctPredictions = [],[]

        # ML model HoeffdingTree
        self.model = preprocessing.StandardScaler() | tree.HoeffdingTreeClassifier()
        self.accuracy = metrics.Accuracy()
        self.precision = metrics.Precision()
        self.recall = metrics.Recall()
        self.f1Score = metrics.F1()

    def run(self):
        """This is the main entry point and uses the other functions"""
        data = self.loadFile()
        for idx, row in data.iterrows():

            x = {} # extract features
            for feat in self.features:
                x[feat] = row[feat]

            y = row['tgt']
            
            # Predict before training
            y_pred = self.model.predict_one(x)
            
            if y_pred is not None:
                self.accuracy.update(y, y_pred)
                self.precision.update(y, y_pred)
                self.recall.update(y, y_pred)
                self.f1Score.update(y, y_pred)
            
            # update the model with new data
            self.model.learn_one(x, y)

            # to avoid overlapping labels in matplotlib graph
            if y == y_pred:
                self.correctPredictions.append(y_pred)

            else: 
                self.incorrectPredictions.append(y)

            # print progress and current metrics every row
           
            print("\n")
            print(f"Processed {idx} samples")
            print(f"Accuracy: {100 * self.accuracy.get():.2f}")
            print(f"Precision: {100 * self.precision.get():.2f}")
            print(f"Recall: {100 * self.recall.get():.2f}")
            print(f"F1 score: {100 * self.f1Score.get():.2f}")

        self.visualiseData()
    
    def visualiseData(self):
        """this function calclates the ratio of correct to incorrect and plots the data"""
        correct_len = len(self.correctPredictions)
        incorrect_len = len(self.incorrectPredictions)

        # calculate the ratio of positives for each negative result
        if incorrect_len == 0:
            ratio = float('inf')  # Avoid division by zero
        else:
            ratio = correct_len / incorrect_len

        plt.figure(figsize=(12,6))
        plt.plot(self.incorrectPredictions, color="red", label='incorrect results')
        plt.plot(self.correctPredictions, color="green", label='correct results', alpha=0.7)
        plt.legend()
        plt.title(f"Actual vs Predicted, rato correct to false = {ratio:.2f}:1  // f1 score = {self.f1Score}")
        plt.xlabel("Sample index (time)")
        plt.ylabel("Target (Up=1, Down=0)")
        plt.show()

    def loadFile(self):
        """ loads file with pandas and drop rows with missing data - if file is missing downloads from github"""

        if not os.path.exists(self.file):
            print("data csv not found downloading")
            data = pandas.read_csv("https://raw.githubusercontent.com/datasets/s-and-p-500/main/data/data.csv")
            data.to_csv(self.file, index=False)
        else:pass

        data = pandas.read_csv(self.file) 
        data['tgt'] = (data["Real Price"].shift(-1) > data["Real Price"]).astype(int)
        data = data.dropna()
        return data 
    
if __name__ == "__main__":
    AI = financialAI(dataFile="main/data.csv")
    AI.run()
