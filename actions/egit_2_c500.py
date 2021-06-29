from src.dataset.dataset4BugPrediction import Dataset4BugPrediction
from src.model.RF4BugPrediction import RF4BugPrediction
from src.result.result4BugPrediction import Result4BugPrediction
from src.manager import Maneger
import datetime
import os
import shutil

class Experiment():
    def __init__(self):
        self.id = os.path.splitext(os.path.basename(__file__))[0]
        print(self.id)
        nameProject, releaseID, period = self.id.split("_")
        print(nameProject)
        print(releaseID)
        print(period)

        self.purpose = ["searchHyperParameter", "searchParameter", "test"]

        self.dataset = Dataset4BugPrediction()
        self.dataset.setPathsRecords4Train(
            [
                r"C:\Users\login\data\workspace\MLTool\datasets\egit\R" + str(int(releaseID)-1) + "_" +period+".csv"
            ]
        )
        self.dataset.setIsCrossValidation(True)
        self.dataset.setSplitSize4Validation(5)
        self.dataset.setPathsRecords4Test(
            [
                r"C:\Users\login\data\workspace\MLTool\datasets\egit\R" + releaseID + "_" +period+"_test.csv"
            ]
        )

        self.model = RF4BugPrediction()
        #self.model.setPeriod4HyperParameterSearch(60*60)
        self.model.setTrials4HyperParameterSearch(100)
        self.model.setIsCrossValidation(True)

        Result4BugPrediction.setPathResult("results/"+self.id)#${result}=results
        os.makedirs(Result4BugPrediction.pathResult, exist_ok=True)
        shutil.copy(__file__, Result4BugPrediction.pathResult)


experiment = Experiment()
maneger = Maneger()
maneger.run(experiment)