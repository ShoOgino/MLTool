from src.manager import MLManeger
from src.utility import UtilPath
import datetime
import json
import re

option={
    "purpose"                    : ["searchHyperParameter", "searchParameter", "test"], #set of "searchHyperParameter", "searchParameter" and "test"
    #"time4searchHyperParameter"  : 60*10, # required if purpose contains "searchHyperParameter"
    "trials4searchHyperParameter": 100000, # required if purpose contains "searchHyperParameter"
    "modelAlgorithm"             : "RF",  # SVM or RF or DNN
    "processor"                  : "CPU", # CPU or GPU
    "pathDatasetDir"             : "egit/datasets/2", # MLTool/datasets/${pathDatasetDir}
    "pathHyperParameter"         : "",    # required if purpose is ["searchParameter"]
    "pathParameter"              : ""    # required if purpose is ["test"]
}
maneger = MLManeger(option)
maneger.act()