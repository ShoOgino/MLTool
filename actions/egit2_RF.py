from src.dataset.dataset4BugPrediction import Dataset4BugPrediction
from src.model.RF4BugPrediction import RF4BugPrediction
from src.result.result4BugPrediction import Result4BugPrediction
from src.manager import Maneger
import datetime
import os
import shutil

class Experiment():
    def __init__(self):
        self.id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        print(self.id)

        self.purpose = ["test"]

        self.dataset = Dataset4BugPrediction()
        self.dataset.setPathDataset4Train(
            [
                "/home/shoogino/workspace/MLTool/datasets/egit/datasets/old/1.csv",
            ]
        )
        self.dataset.setTransform4Train(oversampling=True)  #oversamplingとか正規化とか標準化とか
        self.dataset.setSplitSize4CrossValidation(5)
        self.dataset.setPathDataset4Test("/home/shoogino/workspace/MLTool/datasets/egit/datasets/old/2.csv")
        self.dataset.setTransform4Test(oversampling=True)  #oversamplingとか正規化とか標準化とか

        self.model = RF4BugPrediction()
        #self.model.setPeriod4HyperParameterSearch(60*60)
        self.model.setTrials4HyperParameterSearch(100)
        self.model.setIsCrossValidation(True)

        Result4BugPrediction.setPathResult("results/"+self.id)#${result}=results
        Result4BugPrediction.setPathParameter("parameter")
        os.makedirs(Result4BugPrediction.pathResult, exist_ok=True)
        shutil.copy(__file__, Result4BugPrediction.pathResult)


experiment = Experiment()
maneger = Maneger()
maneger.run(experiment)