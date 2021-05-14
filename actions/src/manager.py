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
    def __init__(self, option):
        self.option = option
        print(json.dumps(self.option,indent=4))
        self.idAction = \
            option["pathDatasetDir"].replace("/","_") + "_" \
            + option["modelAlgorithm"] + "_" \
            + str(option["purpose"]) + "_" \
            + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        print(self.idAction)

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
            #クロスバリデーションを実装する。
            #具体的には、トレーニングデータとバリデーションデータのセットの配列を用意して、それをmodelerに与える。
            #モデラーは、ハイパーパラメータごとに、配列分だけ試行して、その予測精度の平均値を最終的に出力するようにする。
            dataset4tuningHP = []
            for i in range(5):
                dataset={}
                _, dataset["xTrain"], dataset["yTrain"] = datasetLoader.loadTrain4Search(i)
                _, dataset["xValid"], dataset["yValid"] = datasetLoader.loadValid4Search(i)
                dataset4tuningHP.append(dataset)
            self.option["pathHyperParameter"] = modeler.searchHyperParameter(dataset4tuningHP, self.option["modelAlgorithm"], self.option["trials4searchHyperParameter"])
        if "searchParameter" in self.option["purpose"]:
            print("-----searchParameter-----")
            _, xTrain, yTrain = datasetLoader.loadTrain4Test()
            _, xTest , yTest = datasetLoader.loadTest4Test()
            self.option["pathParameter"] = modeler.searchParameter(xTrain, yTrain, xTest, yTest, self.option["modelAlgorithm"], self.option["pathHyperParameter"])
        if "test" in self.option["purpose"]:
            print("-----test-----")
            IDRecord, xTest, yTest = datasetLoader.loadTest4Test()
            modeler.test(IDRecord, xTest, yTest, self.option["modelAlgorithm"], self.option["pathParameter"])