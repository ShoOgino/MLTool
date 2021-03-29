from src.manager import MLManeger
from src.utility import UtilPath
import datetime
import json
import re

option={
    "purpose"                    : ["searchHyperParameter", "searchParameter", "test"], #set of "searchHyperParameter", "searchParameter" and "test"
    "time2searchHyperParameter"  : 30,    # required if purpose contains "searchHyperParameter"
    "modelAlgorithm"             : "RF",  # RF or DNN
    "processor"                  : "CPU", # CPU or GPU
    "pathDatasetDir"             : "egit/isBuggy_moreMetrics_all/2", # MLTool/datasets/${pathDatasetDir}
    "pathHyperParameter"         : "",    # required if purpose is ["searchParameter"]
    "pathParameter"              : "",    # required if purpose is ["test"]
    "date"                       : datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
}
idAction = \
        option["pathDatasetDir"].replace("/","_") + "_" \
        + option["modelAlgorithm"] + "_" \
        + str(option["purpose"]) + "_" \
        + option["date"]

print(json.dumps(option,indent=4))
print(idAction)
maneger = MLManeger(idAction, option)
maneger.act()