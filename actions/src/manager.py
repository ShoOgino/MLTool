import sys
import glob
import pandas as pd
import json
import subprocess
import torch.nn.functional as F
import torch.nn as nn
import torch
from src.result.result4BugPrediction import Result4BugPrediction

class Maneger:
    def __init__(self):
        pass

    def run(self, experiment):
        experiment.dataset.loadRecords4Train()
        experiment.dataset.loadRecords4Test()
        experiment.dataset.standardize()
        experiment.dataset.showSummary()
        if "searchHyperParameter" in experiment.purpose:
            print("-----searchHyperParameter-----")
            pathHyperParameter = experiment.model.searchHyperParameter(
                experiment.dataset.getDataset4SearchHyperParameter()
            )
            Result4BugPrediction.setPathHyperParameter(pathHyperParameter)
        if "searchParameter" in experiment.purpose:
            print("-----searchParameter")
            pathParameter = experiment.model.searchParameter(
                experiment.dataset.getDataset4SearchParameter()
            )
            Result4BugPrediction.setPathParameter(pathParameter)
        if "test" in experiment.purpose:
            print("-----test-----")
            experiment.model.test(
                experiment.dataset.getDataset4Test()
            )