import glob
import argparse
import json

def calcRecallPrecision(pathDir):
    pathTarget = pathDir+"/**/*.json"
    print(pathTarget)
    pathsSample = glob.glob(pathTarget, recursive=True)
    print(len(pathsSample))
    countHasBeenBuggy = 0
    countIsBuggy = 0
    countHBBCaptureIsBuggy = 0
    for pathSample in pathsSample:
        with open(pathSample, encoding="utf-8") as f:
            sample = json.load(f)
            if(sample["isBuggy"]==1):
                countIsBuggy+=1
            if(sample["hasBeenBuggy"]==1):
                countHasBeenBuggy+=1
            if(sample["isBuggy"]==1 and sample["hasBeenBuggy"]==1):
                countHBBCaptureIsBuggy+=1
    print("countIsBuggy: "+str(countIsBuggy))
    print("countHasBeenBuggy: "+str(countHasBeenBuggy))
    print("countHBBCaptureIsBuggy: "+str(countHBBCaptureIsBuggy))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pathDir', type=str)
    args = parser.parse_args()
    calcRecallPrecision(args.pathDir)