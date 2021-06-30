from ntpath import join
from src.dataset.dataset4BugPrediction import Dataset4BugPrediction
from src.model.RF4BugPrediction import RF4BugPrediction
from src.result.result4BugPrediction import Result4BugPrediction
from src.manager import Maneger

import os
import glob
import shutil
from multiprocessing import Pool
from multiprocessing import Process
import multiprocessing

class Experiment():
    def __init__(self, setting):
        self.id = setting["nameProject"]+"_"+setting["numOfRelease"]+"_"+setting["pattern"]

        self.purpose = ["searchHyperParameter", "searchParameter", "test"]

        self.dataset = Dataset4BugPrediction()
        self.dataset.setPathsRecords4Train(
            [
                os.path.join(setting["dirDatasets"], setting["numOfRelease"] + "_" + setting["pattern"] +"_train.csv")
            ]
        )
        self.dataset.setIsCrossValidation(True)
        self.dataset.setSplitSize4Validation(5)
        self.dataset.setPathsRecords4Test(
            [
                os.path.join(setting["dirDatasets"], setting["numOfRelease"] + "_" + setting["pattern"] +"_test.csv")
            ]
        )

        self.model = RF4BugPrediction()
        self.model.setPeriod4HyperParameterSearch(60*60*10)
        #self.model.setTrials4HyperParameterSearch(1)
        self.model.setIsCrossValidation(True)

        Result4BugPrediction.clear()#${result}=results
        Result4BugPrediction.setPathResult("results/"+self.id)#${result}=results
        os.makedirs(Result4BugPrediction.pathResult, exist_ok=True)
        shutil.copy(__file__, Result4BugPrediction.pathResult)

def do(setting):
    print(setting)
    experiment = Experiment(setting)
    maneger = Maneger()
    maneger.run(experiment)

def main():
    dirDatasets = r"../datasets"
    namesProject = [
        "cassandra",
        "checkstyle",
        "egit",
        "jgit",
        "linuxtools",
        "realm-java",
        "sonar-java",
        "poi"
    ]
    settings = []
    for nameProject in namesProject:
        pathsDataset = glob.glob(dirDatasets+"/"+nameProject+"/output/*.csv")
        for pathDataset in pathsDataset:
            filenameDataset = os.path.splitext(os.path.basename(pathDataset))[0]
            setting = {}
            setting["nameProject"] = nameProject
            setting["numOfRelease"] = filenameDataset.split("_")[0]
            setting["pattern"] = filenameDataset.split("_")[1]
            setting["dirDatasets"] = os.path.dirname(pathDataset)
            if(setting["pattern"]=="r" and not setting in settings): settings.append(setting)
    print(settings)
    numOfProcessers = multiprocessing.cpu_count()
    pool = Pool(numOfProcessers -2)
    result = pool.map(do, settings)
#    for setting in settings:
#        do(setting)


if __name__ == '__main__':
    main()