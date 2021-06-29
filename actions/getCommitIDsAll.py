from os import path
from pydriller import RepositoryMining
from datetime import date, timedelta
import csv
import copy
from dateutil.relativedelta import relativedelta
import argparse

metadata = {}

#  cassandra
metadata["cassandra"] = {}
metadata["cassandra"]["pathRepositoryFile"] = r"C:\Users\login\data\workspace\MLTool\datasets\cassandra\repositoryFile"
metadata["cassandra"]["pathRepositoryMethod"] = r"C:\Users\login\data\workspace\MLTool\datasets\cassandra\repositoryMethod"
metadata["cassandra"]["releaseIDs"] = [
    "1f91e99223b0d1b7ed8390400d4a06ac08e4aa85",

    "533193a7b82c98814567c735f13a0e33c58424b1",
    "03045ca22b11b0e5fc85c4fabd83ce6121b5709b",
    "96f407bce56b98cd824d18e32ee012dbb99a0286",

    "e848d47ed171f20ccd8cf5e20d9e188ede85c17c"
]

# checkstyle
metadata["checkstyle"] = {}
metadata["checkstyle"]["pathRepositoryFile"] = r"C:\Users\login\data\workspace\MLTool\datasets\checkstyle\repositoryFile"
metadata["checkstyle"]["pathRepositoryMethod"] = r"C:\Users\login\data\workspace\MLTool\datasets\checkstyle\repositoryMethod"
metadata["checkstyle"]["releaseIDs"] = [
    "5f53a2a380e038e791af244b42f245b813dbc379",

    None,
    "9acdd1b97e561748ae8cc61e11dcc145885ecb6d",
    "cbcc08934f6e1687d55a1174905f7ce95a3ab2c4",
    "00af541f485656d25ceb07e206e82af35847b77a",
    "f31928524128c0909347456ef33105e63ee59824",
    "a780f92fe771d1c062408633eb06453e674b2f2e",
    "635139a14f3db6da443764c62d9c93af0a4e2cd7",
    "f45f6b1917b18d5d5151da5d3f74335934b93d72",

    "8326f7bc6af02b2303b0dc136323e129893da705"
]


#  egit
metadata["egit"] = {}
metadata["egit"]["pathRepositoryFile"] = r"C:\Users\login\data\workspace\MLTool\datasets\egit\repositoryFile"
metadata["egit"]["pathRepositoryMethod"] = r"C:\Users\login\data\workspace\MLTool\datasets\egit\repositoryMethod"
metadata["egit"]["releaseIDs"] = [
    "dfbdc456d8645fc0c310b5e15cf8d25d8ff7f84b",

    "3104c7a3af566617c32172f956eb57d0b135d3bf",
    "1f07f085d6bcae0caf372fffec19583ac5615d3b",
    "f85dc228e3e67b229c978f751a8723a5c810738b",
    "48712bdfa849e1afd0071b937f0b2bef04767610",
    "ba0bcbfe69b2803c843c19b6e800b2525b227cb2",

    "08c530ec099610334a4794f7fe7ed33cc619ef4f"
]

# geotools
metadata["geotools"] = {}
metadata["geotools"]["pathRepositoryMethod"] = r"C:\Users\login\data\workspace\MLTool\datasets\geotools\repositoryMethod"
metadata["geotools"]["pathRepositoryFile"] = r"C:\Users\login\data\workspace\MLTool\datasets\geotools\repositoryFile"
metadata["geotools"]["releaseIDs"] = [
    "84d6dc79f544430bdb6bdec7fa3a7fb89bb65229",

    None,
    None,
    None,
    None,
    None,
    None,
    None,
    "fddfd9bed8fbea87479602df7643ab44298723da",
    "cd6a0ac13e0ccd9ed3df653e1d7ca640a80ed86d",
    "115fd1004418702455df7bae10e90f7c14d8fa64",

    "45103e4ea4505d38404ddf0ab0c8d67cdac4f3cb",
    "645963cd65a6f5c151722ce012f2a4a6d3b86a44",
    "3e400364d95c06a3e449c34dbc3c587a12bbdfb6",
    "19819635d08697eaf28356e0fa8d084c77143387",
    "d18ecd3d875a9136f2644b29d1f28f7ec67a06db",
    "ae16e116c58d9f4bc3fcec2566fca3dd8dd92120",
    "cf34347a4544cac900c09f2e5de4c5fb138c8637",
    "91315ffe07add4d70567a36802e5813e9cc75fa0",
    "92b83e521e75c852e3c072fccce0200c9bd2fcde",
    "ffc271f317c04e714ea44a4879dd4601bd723d5e",

    "90b0c1c482733f6914bbf10aad19e78824634a69",
    "2f654c50828efc8f42b23e1363a25fafd4444de3",
    "e4e8e203d1612d8d9b14bdace757d0bd79163be2",
    "81c98ca8040abb325a17c85950645ce89bd93088",
    "d187663948b47212514991a71c0fc3021908b034",

    "81e3f91bb97fd2a8a7b4b2609aba40c9ba8ab9ae"
]



#   jgit
metadata["jgit"] = {}
metadata["jgit"]["pathRepositoryFile"] = r"C:\Users\login\data\workspace\MLTool\datasets\jgit\repositoryFile"
metadata["jgit"]["pathRepositoryMethod"] = r"C:\Users\login\data\workspace\MLTool\datasets\jgit\repositoryMethod"
metadata["jgit"]["releaseIDs"] = [
    "1a6964c8274c50f0253db75f010d78ef0e739343",

    "b26ff6ebd623f7f3bf9b099ce10330d2964cc33b",
    "aacd4f721bba97eda3756e2fcbd71b5e57b3673a",
    "f384644774ae01823e850c862aeda5bddb4a4326",
    "4f221854556991e3394b3a71e77ee0b771b1500b",
    "e729a83bd24bbc25f7ac209baee01f561fe218c8",

    "91b2e167a2341d41212ea801943fb82a148e3b7f"
]


#   linuxtools
metadata["linuxtools"] = {}
metadata["linuxtools"]["pathRepositoryFile"] = r"C:\Users\login\data\workspace\MLTool\datasets\linuxtools\repositoryFile"
metadata["linuxtools"]["pathRepositoryMethod"] = r"C:\Users\login\data\workspace\MLTool\datasets\linuxtools\repositoryMethod"
metadata["linuxtools"]["releaseIDs"] = [
    "e3ea1e9cc32a55b79c2735f36a07c90975bec67d",

    "b1172eebee348db6cd979533f0d38a67725c7da8",
    "2aa4892cece1fc393f8bec3083a6303912d17f16",
    "38b6b6787381da5d46cb9006c451a2086c64a4d2",
    "e3de351ca7ffd1bf722c9f0e2e86998cf1160bae",
    "4d86c7a081b74595b925ffa64c9578e36b0f7374",
    "fc1655d0c62b3ea7f2695f49c0f159d762bb4f27",
    "1c9cc481d11079740a3d309d7bcb9209757feaab",
    "65dafd6be0291db5a63f1269184bfde54518c7c4",

    "7ff527db3a1f93059236c270e7566e6a97ffe380"
]



#   lucene-solr
metadata["lucene-solr"] = {}
metadata["lucene-solr"]["pathRepositoryFile"] = r"C:\Users\login\data\workspace\MLTool\datasets\lucene-solr\repositoryFile"
metadata["lucene-solr"]["pathRepositoryMethod"] = r"C:\Users\login\data\workspace\MLTool\datasets\lucene-solr\repositoryMethod"
metadata["lucene-solr"]["releaseIDs"] = [
    "a0e7ee9d0d12370e8d2b5ae0a23b6e687e018d85",
    
    None,
    "0e45934a35390eb2c370f2b6169f4bd23968d324",
    "6dbb4c2facc86eee76c97e58398e3364ff39ceb7",
    "30b22298bacb5238437864128bc9005d49d0d3ed",
    "859fdefbceb3ffe088e5091fd695380e81a49c11",
    "48c80f91b8e5cd9b3a9b48e6184bd53e7619e7e3",
    "3ba304b29825a94249c5145b3f5061e87b87d8f8",
    "2ae4746365c1ee72a0047ced7610b2096e438979",

    "180058fe7897923d9eb150039a325faf0def3f61"
]


#   poi
metadata["poi"] = {}
metadata["poi"]["pathRepositoryFile"] =r"C:\Users\login\data\workspace\MLTool\datasets\poi\repositoryFile"
metadata["poi"]["pathRepositoryMethod"] = r"C:\Users\login\data\workspace\MLTool\datasets\poi\repositoryMethod"
metadata["poi"]["releaseIDs"] = [
    "6c234ab6c8d9b392bc93f28edf13136b3c053894",

    None,
    "0dd67eac90ac556235d5bfa1ec9e4e1265f30931",
    "2e335762bd306da833514d3a584c5f28ef8eb184",
    "333b47391399898dca8e91366d316c5a8aeb1809",

    "382714eccd92667fc83f70115b736c64ebff9700"
]


# realm-java
metadata["realm-java"] = {}
metadata["realm-java"]["pathRepositoryFile"] = r"C:\Users\login\data\workspace\MLTool\datasets\realm-java\repositoryFile"
metadata["realm-java"]["pathRepositoryMethod"] = r"C:\Users\login\data\workspace\MLTool\datasets\realm-java\repositoryMethod"
metadata["realm-java"]["releaseIDs"] = [
    "b03c621431fdc7e6e43566eeef505a32f5f6ce83",

    "eef492341a17351dc671b296bf7f1cd2c2ed32a4",
    "1f19a05f820c1e43a9a0e38d8e32b0d96920df7f",
    "66fb375b32b7db660c1d06fc7e27bf708d8cebab",
    "e26255b9c5248620861565eb3caf7c908c4f8277",
    "8740eb6ce5bfc8536be2b480988a6212f2ce8466",
    "8a022573a5b095c2ec887720bd73098475829766",
    "5e1cb707bf37a4c5fa03b45cd5a4fde39fed146d",

    "fd3446d65856fc46bebf1e0632dca3b8260e2d12"
]


# sonar-java
metadata["sonar-java"] = {}
metadata["sonar-java"]["pathRepositoryFile"] = r"C:\Users\login\data\workspace\MLTool\datasets\sonar-java\repositoryFile"
metadata["sonar-java"]["pathRepositoryMethod"] = r"C:\Users\login\data\workspace\MLTool\datasets\sonar-java\repositoryMethod"
metadata["sonar-java"]["releaseIDs"] = [
    "5393bd3cc71e716dcd946232f5d43297d5889357",

    "5ac4cf695248bc7385cb3377216cd86340bda0b0",
    "fb4e10dcf17447ecea699934cdef0cf2009b1a52",
    "65396a609ddface8b311a6a665aca92a7da694f1",
    "b653c6c8640ab3d6015d036a060f58e027a653af",
    "fe9584c9f7812edc46f32d29440fc81b85a597a4",
    "931433c0510b161974d4844679f7bf3c73bb3e37",

    "a4121b68b68740704a816ef7ad6636b61ab7ca55"
]


#   wicket
metadata["wicket"] = {}
metadata["wicket"]["pathRepositoryFile"] = r"C:\Users\login\data\workspace\MLTool\datasets\wicket\repositoryFile"
metadata["wicket"]["pathRepositoryMethod"] = r"C:\Users\login\data\workspace\MLTool\datasets\wicket\repositoryMethod"
metadata["wicket"]["releaseIDs"] = [
    None,

    None,
    None,
    None,
    None,
    None,
    "0d75ee57abb31b4db48c0396870aba39c4d9ddee",
    "98a3a6295f426aa25a121f914a41bf792df8fdb0",
    "5e789f1c98f6d57dba17f896c6220b0202af08a9",
    "c2802f3ef8df9833da63d144fb4ad03d59e31acc",

    "048e5681df960018722237ee6254273e839b5fd2"
]



def getCorrespondingCommit(idCommitFile):#file->method
    print(idCommitFile)
    for commit in RepositoryMining(pathRepositoryFile, include_refs=True).traverse_commits():
        if(commit.hash==idCommitFile):
            dateCommitUntil = commit.committer_date
    idCommit = ""
    dateCommit = None
    for commit in RepositoryMining(pathRepositoryMethod, include_refs=True).traverse_commits():
        if(
            (dateCommit == None and commit.committer_date <= dateCommitUntil) or
            (dateCommit != None and dateCommit <= commit.committer_date and commit.committer_date <= dateCommitUntil)
        ):
            idCommit = commit.hash
            dateCommit = commit.committer_date
    return idCommit

records = []
def identifyCommit_rbr(releaseIndex):
    if(releaseIDs[releaseIndex-1]==None):
        return None
    if(releaseIDs[releaseIndex-2]==None):
        return None
    testCommitID          = releaseIDs[releaseIndex]
    trainCommitID         = releaseIDs[releaseIndex-1]
    previousTrainCommitID = releaseIDs[releaseIndex-2]
    if(releaseIndex-2<0 or previousTrainCommitID==None or trainCommitID ==None):
        return None
    test  = [
        "R"+str(releaseIndex)+"_r_test",
        "method",
        "metrics_isBuggy_hasBeenBuggy",
        trainCommitID,
        testCommitID,
        releaseIDs[len(releaseIDs)-1],
        getCorrespondingCommit(trainCommitID),
        getCorrespondingCommit(testCommitID),
        getCorrespondingCommit(releaseIDs[len(releaseIDs)-1])
    ]
    records.append(test)
    train = [
        "R"+str(releaseIndex)+"_r_train",
        "method",
        "metrics_isBuggy_hasBeenBuggy",
        previousTrainCommitID,
        trainCommitID,
        testCommitID,
        getCorrespondingCommit(previousTrainCommitID),
        getCorrespondingCommit(trainCommitID),
        getCorrespondingCommit(testCommitID)
    ]
    records.append(train)
def identifyCommit_cbc(commitHashes, releaseIndex, NumOfCommits):
    commitIDsTest = [None, None, None]
    commitIDsTest[1] = releaseIDs[releaseIndex]
    if(commitHashes.index(commitIDsTest[1])-NumOfCommits*2<0):
        return None
    commitIDsTest[0] = commitHashes[commitHashes.index(commitIDsTest[1])-NumOfCommits]
    commitIDsTest[2] = releaseIDs[len(releaseIDs)-1]
    test  = [
        "R"+str(releaseIndex)+"_c"+str(NumOfCommits) + "_test",
        "method",
        "metrics_isBuggy_hasBeenBuggy",
        commitIDsTest[0],
        commitIDsTest[1],
        commitIDsTest[2],
        getCorrespondingCommit(commitIDsTest[0]),
        getCorrespondingCommit(commitIDsTest[1]),
        getCorrespondingCommit(commitIDsTest[2]),
    ]
    records.append(test)
    count = 0
    while(0 < commitHashes.index(commitIDsTest[1])-NumOfCommits-NumOfCommits*count):
        commitIDsTrain = [None, None, None]
        commitIDsTrain[0] = commitHashes[commitHashes.index(commitIDsTest[1]) - NumOfCommits*2 -NumOfCommits*count]
        commitIDsTrain[1] = commitHashes[commitHashes.index(commitIDsTest[1]) - NumOfCommits   -NumOfCommits*count]
        commitIDsTrain[2] = commitIDsTest[1]
        train = [
            "R"+str(releaseIndex)+"_c"+str(NumOfCommits) + "_train_" + str(count),
            "method",
            "metrics_isBuggy_hasBeenBuggy",
            commitIDsTrain[0],
            commitIDsTrain[1],
            commitIDsTrain[2],
            getCorrespondingCommit(commitIDsTrain[0]),
            getCorrespondingCommit(commitIDsTrain[1]),
            getCorrespondingCommit(commitIDsTrain[2])
        ]
        records.append(train)
        count+=1
def identifyCommit_tbt(commits, releaseIndex, numOfMonths):
    commitIDsTest = [None, None, None]
    commitIDsTest[1] = releaseIDs[releaseIndex]
    commitIDsTest[2] = releaseIDs[len(releaseIDs)-1]
    testSince = commits[commitHashes.index(commitIDsTest[1])].committer_date - relativedelta(months = numOfMonths)
    testUntil = commits[commitHashes.index(commitIDsTest[1])].committer_date
    for commit in reversed(commits):
        if(testSince<=commit.committer_date and commit.committer_date<=testUntil):
            commitIDsTest[0] = commit.hash
    test  = [
        "R"+str(releaseIndex)+"_t"+str(numOfMonths) + "_test",
        "method",
        "metrics_isBuggy_hasBeenBuggy",
        commitIDsTest[0],
        commitIDsTest[1],
        commitIDsTest[2],
        getCorrespondingCommit(commitIDsTest[0]),
        getCorrespondingCommit(commitIDsTest[1]),
        getCorrespondingCommit(commitIDsTest[2])
    ]
    records.append(test)

    count = 0
    while(True):
        commitIDsTrain = [None, None, None]
        intervals = [
            [commits[commitHashes.index(commitIDsTest[1])].committer_date - relativedelta(months = numOfMonths*3 + numOfMonths*count), commits[commitHashes.index(commitIDsTest[1])].committer_date - relativedelta(months = numOfMonths*2 + numOfMonths*count)],
            [commits[commitHashes.index(commitIDsTest[1])].committer_date - relativedelta(months = numOfMonths*2 + numOfMonths*count), commits[commitHashes.index(commitIDsTest[1])].committer_date - relativedelta(months = numOfMonths*1 + numOfMonths*count)],
        ]
        for i in range(len(intervals)):
            for commit in reversed(commits):
                if(intervals[i][0] < commit.committer_date and commit.committer_date <= intervals[i][1]):
                    commitIDsTrain[i] = commit.hash
                    break
        commitIDsTrain[2] = commitIDsTest[1]
        if(commitIDsTrain[0]==None or commitIDsTrain[1]==None or commitIDsTrain[2]==None):
            if(commits[commitHashes.index(commitIDsTest[1])].committer_date - relativedelta(months = numOfMonths*1 + numOfMonths*count) < commits[0].committer_date):
                break
            else:
                count+=1
                continue
        train = [
            "R"+str(releaseIndex)+"_t"+str(numOfMonths) + "_train_" + str(count),
            "method",
            "metrics_isBuggy_hasBeenBuggy",
            commitIDsTrain[0],
            commitIDsTrain[1],
            commitIDsTrain[2],
            getCorrespondingCommit(commitIDsTrain[0]),
            getCorrespondingCommit(commitIDsTrain[1]),
            getCorrespondingCommit(commitIDsTrain[2])
        ]
        records.append(train)
        count+=1

pathRepositoryMethod = None
pathRepositoryFile = None
releaseIDs = None
commitHashes = []
def main(project):
    global pathRepositoryMethod
    global pathRepositoryFile
    global releaseIDs
    pathRepositoryMethod = metadata[project]["pathRepositoryMethod"]
    pathRepositoryFile = metadata[project]["pathRepositoryFile"]
    releaseIDs = metadata[project]["releaseIDs"]

    commits = []
    for commit in RepositoryMining(pathRepositoryFile, include_refs=True).traverse_commits():
        commitHashes.append(str(commit.hash))
        commits.append(commit)
    for releaseIndex in range(len(releaseIDs)):
        print("releaseID: "+str(releaseIndex))
        if(
            releaseIndex==0 or
            releaseIndex ==1 or
            releaseIndex == len(releaseIDs)-1 or
            releaseIDs[releaseIndex]==None
        ):
            continue
        print("numRelease: "+ str(releaseIndex))
        identifyCommit_rbr(releaseIndex)
        numOfCommitss = [500, 1000, 1500, 2000, 2500]
        for i, numOfCommits in enumerate(numOfCommitss):
            print("numOfCommits: "+ str(numOfCommits))
            identifyCommit_cbc(commitHashes, releaseIndex, numOfCommits)
        numOfMonthss = [1, 2, 3, 6, 12]
        for i, numOfMonths in enumerate(numOfMonthss):
            print("numOfMonths: "+ str(numOfMonths))
            identifyCommit_tbt(commits, releaseIndex, numOfMonths)
    with open(project+'.csv', 'w') as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(records)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""Convert a git log output to json.
                                                 """)
    parser.add_argument('--project', type=str)

    args = parser.parse_args()
    project = args.project
    main(project)
