from src.dataset.dataset4BugPrediction import Dataset4BugPrediction
from src.model.DNN4BugPrediction_Pytorch import DNN4BugPrediction_Pytorch
from src.result.result4BugPrediction import Result4BugPrediction
from src.manager import Maneger
import datetime
import os
import shutil

class Experiment():
    def __init__(self):
        self.id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        print(self.id)

        self.purpose = ["searchHyperParameter", "searchParameter", "test"]

        self.dataset = Dataset4BugPrediction()
        self.dataset.setPathsRecords4Train(
            [
                r"C:\Users\login\data\workspace\MLTool\datasets\egit\isBuggy\2\train0.csv"
            ]
        )
        self.dataset.setIsCrossValidation(True)
        self.dataset.setSplitSize4Validation(5)
        self.dataset.setPathsRecords4Test(
            [
                r"C:\Users\login\data\workspace\MLTool\datasets\egit\isBuggy\2\test.csv"
            ]
        )

        self.model = DNN4BugPrediction_Pytorch()
        #self.model.setPeriod4HyperParameterSearch(60*60)
        self.model.setTrials4HyperParameterSearch(100)
        self.model.setIsCrossValidation(True)

        Result4BugPrediction.setPathResult("results/"+self.id)#${result}=results
        os.makedirs(Result4BugPrediction.pathResult, exist_ok=True)
        shutil.copy(__file__, Result4BugPrediction.pathResult)


experiment = Experiment()
maneger = Maneger()
maneger.run(experiment)