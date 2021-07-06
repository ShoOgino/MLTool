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
import pprint
import math
import re
import operator

class Experiment():
    def __init__(self, setting):
        self.id = setting["nameProject"]+"_"+setting["numOfRelease"]+"_"+setting["interval"]+"_"+str(setting["torikata"])

        self.purpose = ["searchHyperParameter", "searchParameter", "test"]

        self.dataset = Dataset4BugPrediction()
        self.dataset.setPathsRecords4Train(setting["pathsDataset4Train"])
        self.dataset.setIsCrossValidation(True)
        self.dataset.setSplitSize4Validation(5)
        self.dataset.setPathsRecords4Test(setting["pathsDataset4Test"])

        self.model = RF4BugPrediction()
        self.model.setPeriod4HyperParameterSearch(10)
        #self.model.setTrials4HyperParameterSearch(1)
        self.model.setIsCrossValidation(True)

        Result4BugPrediction.clear()#${result}=results
        Result4BugPrediction.setPathResult("results/"+self.id)#${result}=results
        os.makedirs(Result4BugPrediction.pathResult, exist_ok=True)
        shutil.copy(__file__, Result4BugPrediction.pathResult)

def do(setting):
    experiment = Experiment(setting)
    maneger = Maneger()
    maneger.run(experiment)

def main():
    dirDatasets = r"../datasets"
    #namesProject = [
    #    "cassandra",
    #    "checkstyle",
    #    "egit",
    #    "jgit",
    #    "linuxtools",
    #    "realm-java",
    #    "sonar-java",
    #    "poi",
    #    "wicket"
    #]
    namesProject = [
        "cassandra"
    ]
    patterns4DatasetTrain = [
        1,
        2,
        3,
        4,
        5
    ]
    settings = []
    patterns = []
    for nameProject in namesProject:
        pathsDataset = glob.glob(dirDatasets+"/"+nameProject+"/output/*.csv")
        for pathDataset in pathsDataset:
            filenameDataset = os.path.splitext(os.path.basename(pathDataset))[0]
            pattern = {}
            pattern["nameProject"] = nameProject
            pattern["numOfRelease"] = filenameDataset.split("_")[0]
            pattern["interval"] = filenameDataset.split("_")[1]
            if(not pattern in patterns): patterns.append(pattern)
    for pattern in patterns:
        for torikata in patterns4DatasetTrain:
            setting = {}
            setting["nameProject"] = pattern["nameProject"]
            setting["numOfRelease"] = pattern["numOfRelease"]
            setting["interval"] = pattern["interval"]
            setting["torikata"] = torikata
            expressionDataset4Train = dirDatasets+"/"+setting["nameProject"]+"/output/"+pattern["numOfRelease"]+"_"+pattern["interval"]+"_"+"train*.csv"
            pathsAll = glob.glob(expressionDataset4Train)
            pathsAll = sorted(pathsAll, key=lambda s: int(re.findall(r'\d+', s)[2]))
            #pprint.pprint(pathsAll)
            setting["pathsDataset4Train"] = pathsAll[0:math.ceil((len(pathsAll)/5)*setting["torikata"])]
            setting["pathsDataset4Test"] = [dirDatasets+"/"+setting["nameProject"]+"/output/"+pattern["numOfRelease"]+"_"+pattern["interval"]+"_"+"test.csv"]
            settings.append(setting)
#    numOfProcessers = multiprocessing.cpu_count()
#    pool = Pool(numOfProcessers -2)
#    result = pool.map(do, settings)
    settings.sort(key=operator.itemgetter('nameProject', 'numOfRelease', 'interval', 'torikata'))
    print(settings)
    for setting in settings:
        pprint.pprint(setting)
        do(setting)


if __name__ == '__main__':
    main()