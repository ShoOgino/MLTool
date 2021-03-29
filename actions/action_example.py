from src.manager import MLManeger
from src.utility import UtilPath
import datetime
import json
import re

option={
    "purpose"                    : ["searchHyperParameter", "searchParameter", "test"], #set of "searchHyperParameter", "searchParameter" and "test"
    "time2searchHyperParameter"  : 60,    # required if purpose contains "searchHyperParameter"
    "modelAlgorithm"             : "DNN",  # RF or DNN
    "processor"                  : "GPU", # CPU or GPU
    "pathDatasetDir"             : "egit/isBuggy/4", # MLTool/datasets/${pathDatasetDir}
    "pathHyperParameter"         : "",    # required if purpose is ["searchParameter"]
    "pathParameter"              : ""    # required if purpose is ["test"]
}
idAction = \
        option["pathDatasetDir"].replace("/","_") + "_" \
        + option["modelAlgorithm"] + "_" \
        + str(option["purpose"]) + "_" \
        + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

print(json.dumps(option,indent=4))
print(idAction)
maneger = MLManeger(idAction, option)
maneger.act()