import os
import sys
import glob
import pandas as pd
import json
import datetime
import subprocess
import tensorflow as tf

from src.modeler import Modeler
from src.utility import UtilPath
from src.datasetLoader import DatasetLoader

class MLManeger:
    def __init__(self, idAction, option):
        self.idAction = idAction
        self.option = option

    def setProcessor(self):
        if self.option["processor"]=="CPU":
            os.environ["CUDA_VISIBLE_DEVICES"]="-1"
        elif self.option["processor"]=="GPU":
            physical_devices = tf.config.experimental.list_physical_devices('GPU')
            if len(physical_devices) > 0:
                for k in range(len(physical_devices)):
                    tf.config.experimental.set_memory_growth(physical_devices[k], True)
                    print('memory growth:', tf.config.experimental.get_memory_growth(physical_devices[k]))
            else:
                raise Exception("enough GPU hardware devices are unavailable.")
        else:
            raise Exception("option \"processor\" must be CPU or GPU.")

    def act(self):
        self.setProcessor()
        os.makedirs(UtilPath.ResultAction(self.idAction), exist_ok=True)
        modeler = Modeler(UtilPath.ResultAction(self.idAction))
        datasetLoader = DatasetLoader(self.option["pathDatasetDir"], self.option["purpose"])
        if "searchHyperParameter" in self.option["purpose"]:
            print("-----searchHyperParameter-----")
            _, xTrain, yTrain = datasetLoader.loadTrain4Search()
            _, xValid, yValid = datasetLoader.loadValid4Search()
            self.option["pathHyperParameter"] = modeler.searchHyperParameter(xTrain, yTrain, xValid, yValid, self.option["modelAlgorithm"], self.option["time2searchHyperParameter"])
        if "searchParameter" in self.option["purpose"]:
            print("-----searchParameter-----")
            _, xTrain, yTrain = datasetLoader.loadTrain4Test()
            _, xTest , yTest = datasetLoader.loadTrain4Test()
            self.option["pathParameter"] = modeler.searchParameter(xTrain, yTrain, xTest, yTest, self.option["modelAlgorithm"], self.option["pathHyperParameter"])
        if "test" in self.option["purpose"]:
            print("-----test-----")
            IDTest, xTest, yTest = datasetLoader.loadTest4Test()
            modeler.test(IDTest, xTest, yTest, self.option["modelAlgorithm"], self.option["pathParameter"])