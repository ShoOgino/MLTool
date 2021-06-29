import glob
import json
import pprint
import os
import csv

records = {
    "c500":[],
    "c1000":[],
    "c1500":[],
    "c2000":[],
    "c2500":[],
    "r":[],
    "t1":[],
    "t2":[],
    "t3":[],
    "t6":[],
    "t12":[]
}
heikin={
    "c500":[],
    "c1000":[],
    "c1500":[],
    "c2000":[],
    "c2500":[],
    "r":[],
    "t1":[],
    "t2":[],
    "t3":[],
    "t6":[],
    "t12":[]
}
pathsResult = glob.glob('results/**/', recursive=True)
pathsResult.sort()
pathsResult.remove("results\\")
for pathResult in pathsResult:
    pathReport = pathResult+"report.json"
    print(pathReport)
    if(os.path.exists(pathReport)):
        with open(pathReport, encoding="utf-8") as f:
            report = json.load(f)
            pattern = os.path.basename(os.path.dirname(pathResult))
            record = [pattern, report["1"]["precision"], report["1"]["recall"], report["1"]["f1-score"], report["AUC"]]
            pprint.pprint(record)
            group = os.path.basename(os.path.dirname(pathResult)).split("_")[2]
            records[group].append(record)
for key in records:
    heikin["key"] = records[key]
with open("records.csv", encoding="utf-8", mode="w", newline="") as f:
    writer = csv.writer(f)
    for key in records:
        writer.writerows(records[key])
