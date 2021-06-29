from pydriller import RepositoryMining
from datetime import date, timedelta
import csv
import copy
from dateutil.relativedelta import relativedelta

#  cassandra
#pathRepositoryFile = r"C:\Users\login\data\workspace\MLTool\datasets\cassandra\repositoryFile"
#pathRepositoryMethod = r"C:\Users\login\data\workspace\MLTool\datasets\cassandra\repositoryMethod"
#releasesIDs = [
#    ["", "56157fc6d98b9a48cfbd2977d8eb757a508ddd61"],
#
#    ["845af1ca3ce21fb543daba5434751cb63fd5a18d", "5c6afc827e5e365b8db887d5d6375bf28f3617c4"],
#    ["5ea3b153840d069b50361f38b8b7050f3c09176b", "723f268a0d42a4d44f3e628db2e17e75e21b64d3"],
#    ["12b8c696bfd8ff9b63bd44dbed1fcfe95bc21aab", "1b723d6a3d5e77f229857e750af67f352a25c5ec"],
#    
#    ["", "6d97484ca948055cd306e35b9b6e760e616cead8"]
#]

# checkstyle
#pathRepositoryFile = r"C:\Users\login\data\workspace\MLTool\datasets\checkstyle\repositoryFile"
#pathRepositoryMethod = r"C:\Users\login\data\workspace\MLTool\datasets\checkstyle\repositoryMethod"
#releasesIDs = [
#    ["", "445ce242aac577adf26950971c14d495df04c5b4"],
#
#    None,
#    ["9acdd1b97e561748ae8cc61e11dcc145885ecb6d", "88699d928c395b6e0b03e21a47e41f660bd7e497"],
#    ["cbcc08934f6e1687d55a1174905f7ce95a3ab2c4", "fa3b16037691ed4d1db4bba19cefbcfb7761c13a"],
#    ["00af541f485656d25ceb07e206e82af35847b77a", "a94afb8484cffc2b72c847ae4e08c2262d15ade7"],
#    ["f31928524128c0909347456ef33105e63ee59824", "1170ec88489233f1642cb7b005256f47c527104a"],
#    ["a780f92fe771d1c062408633eb06453e674b2f2e", "309c632f8fc1c7546878bda8d393b7f272050098"],
#    ["635139a14f3db6da443764c62d9c93af0a4e2cd7", "b68e399a7c8bc61d25dd714ad10ee0a0a06ff735"],
#    ["f45f6b1917b18d5d5151da5d3f74335934b93d72", "bd3d0d7f8a8c339922ea2cdd52f69175302f1edf"],
#
#    ["","e5322c9ce4f4107c0a9531eb457c41be3fb84d93"]
#]


#   eclipse.jdt.core
#pathRepositoryFile = "r"C:\Users\login\data\workspace\MLTool\datasets\eclipse.jdt.core\repositoryFile"
#pathRepositoryMethod = r"C:\Users\login\data\workspace\MLTool\datasets\eclipse.jdt.core\repositoryMethod"
#releasesFile = [
#    "be6c0d208933ac936a6ccb6c66b03d3da13e3796",
#    None,
#    "d0dd6c20e4958b2e8f8c4f7f60f4a15fff6ca500",
#    "c5f94c1be0604760b768518fd2d7014d6ae18052",
#    "3b4f046b547174509da38e0b0a4f7bf6125e51ec",
#    "145fe3a737c63e3d079ddf2fc46dc2640a129635",
#]
#releasesMethod = [
#    "a7fce940d379c2e5e244d9ddaf1acb77c5df6fe5",
#    None,
#    "ba370a9c7ffb041448d1d6f1b3ed0bcf4b2f36f5",
#    "7c3c5b149a5b757dafc9eee38cc2d25bd109519c",
#    "ebc23690a9bf0afac24abe2147261af0fbe9fa10",
#    "4cc601fa3a7ae1b39957dc3e4ff408a522b0c323",
#]


#  egit
pathRepositoryFile = r"C:\Users\login\data\workspace\MLTool\datasets\egit\repositoryFile"
pathRepositoryMethod = r"C:\Users\login\data\workspace\MLTool\datasets\egit\repositoryMethod"
releasesIDs = [
    ["","2c1b0f4ad24fb082e5eb355e912519c21a5e3f41"],

    ["0cc8d32aff8ce91f71d2cdac8f3e362aff747ae7", "1241472396d11fe0e7b31c6faf82d04d39f965a6"],
    ["1f07f085d6bcae0caf372fffec19583ac5615d3b", "2774041935d41453e5080f0e3cbeef136a05597d"],
    ["f85dc228e3e67b229c978f751a8723a5c810738b", "6512ce982bf2376ae7ad308ba3b4a8fafe233376"],
    ["48712bdfa849e1afd0071b937f0b2bef04767610", "5f42c83e73b0681da81e7466ae82830d927e916f"],
    ["ba0bcbfe69b2803c843c19b6e800b2525b227cb2", "c6debf07f32e43470bd310f7b60bc19dd78f2506"],

    ["", "b459d7381ea57e435bd9b71eb37a4cb4160e252b"]
]

# geotools
#pathRepositoryMethod = r"C:\Users\login\data\workspace\MLTool\datasets\geotools\repositoryMethod"
#pathRepositoryFile = r"C:\Users\login\data\workspace\MLTool\datasets\geotools\repositoryFile"
#releasesIDs = [
#    ["", "57dab4b0790c9a2600a004ade2ca44f0f06d8d72"],
#
#    None,
#    None,
#    None,
#    None,
#    None,
#    None,
#    None,
#    ["fddfd9bed8fbea87479602df7643ab44298723da", "d641e96eea6a3d2f05f1d364cc5cbe061c97394d"],
#    ["cd6a0ac13e0ccd9ed3df653e1d7ca640a80ed86d", "cab8f0c92329e837f9f92c8bc4e51a305ad81cd9"],
#    ["115fd1004418702455df7bae10e90f7c14d8fa64", "d0c3a6c296639bb71cb3f30c2115025d9cb4907a"],
#
#    ["45103e4ea4505d38404ddf0ab0c8d67cdac4f3cb", "5d12c5c6f1520526fdaa2fec4e01bdb97cb2b6b1"],
#    ["645963cd65a6f5c151722ce012f2a4a6d3b86a44", "e0a51407284935a6ddaba4631c8289027c3a1f81"],
#    ["3e400364d95c06a3e449c34dbc3c587a12bbdfb6", "fc322919bd85e2817f7fc80d38d2023ea790d6af"],
#    ["19819635d08697eaf28356e0fa8d084c77143387", "0919a3d12e4c7fe61b19b39bc8c255487b0e1860"],
#    ["d18ecd3d875a9136f2644b29d1f28f7ec67a06db", "99d4538285b2c873feddd69a9b7c2987d7ec53f8"],
#    ["ae16e116c58d9f4bc3fcec2566fca3dd8dd92120", "f0abb7cd9526b12f5a2cf26625c1c3fa288b09d9"],
#    ["cf34347a4544cac900c09f2e5de4c5fb138c8637", "7e7c2c70059a4da7d87a9c91a29ecbef6efd3bd8"],
#    ["91315ffe07add4d70567a36802e5813e9cc75fa0", "289581ec6308c34e3d70e5f438fc2262ba45af32"],
#    ["92b83e521e75c852e3c072fccce0200c9bd2fcde", "527573f6f0f90127717a37bc70940932ca7ed7de"],
#    ["ffc271f317c04e714ea44a4879dd4601bd723d5e", "357469ce9b165470980abb741158a17d50338bcc"],
#
#    ["90b0c1c482733f6914bbf10aad19e78824634a69", "6c2125f35f0776ab66e5288fd1bf079a1bc377cc"],
#    ["2f654c50828efc8f42b23e1363a25fafd4444de3", "5e726a787c28ac10749c168d10ac4824c7871b98"],
#    ["e4e8e203d1612d8d9b14bdace757d0bd79163be2", "7bafa37eebdb58afdb46512b5b4038e10db394e9"],
#    ["81c98ca8040abb325a17c85950645ce89bd93088", "9ca218e2cc4bf1eff40d78d535e76e45ae95a395"],
#    ["d187663948b47212514991a71c0fc3021908b034", "c9d85d2362aec14c176ee0d0797e5f99bfaf4aef"],
#
#    ["","4ba1adce1a971ca7e8a127c337996b78fb9355a7"]
#]



#   jgit
#pathRepositoryFile = r"C:\Users\login\data\workspace\MLTool\datasets\jgit\repositoryFile"
#pathRepositoryMethod = r"C:\Users\login\data\workspace\MLTool\datasets\jgit\repositoryMethod"
#releasesIDs = [
#    ["","004bd909df360ac2c699073636da678699853b19"],
#
#    ["f65513f75397b1f4de6d063962e5cccca5a89cb6","a2bc9ac8a1ea177ca25a8410d1daae80ce34e940"],
#    ["aacd4f721bba97eda3756e2fcbd71b5e57b3673a","730797027641e02f9f05f2481fa3b9eddfa90f91"],
#    ["f384644774ae01823e850c862aeda5bddb4a4326","63c202af7960e18aafa54426d66ebb013d72461c"],
#    ["4f221854556991e3394b3a71e77ee0b771b1500b","436e8b67ec6e56eba12d23ab223ed91ea9cf6cd2"],
#    ["e729a83bd24bbc25f7ac209baee01f561fe218c8","35b02784c066ad915030c9182b9ec38945133233"],
#
#    ["","fe7286aec4fd81f3dd765d83c6231728704d5c65"]
#]


#   linuxtools
#pathRepositoryFile = r"C:\Users\login\data\workspace\MLTool\datasets\linuxtools\repositoryFile"
#pathRepositoryMethod = r"C:\Users\login\data\workspace\MLTool\datasets\linuxtools\repositoryMethod"
#releasesIDs = [
#    ["","ad8b5bf026e3d12e499a3f5acf6ea6469261d377"],
#    
#    ["98e11be9f43ee4b4a574fac1437efc73c88878fd","94ead44b7658adb9c528dc2fc3f1cb677669eb46"],
#    ["d20eefcd63bfbb4f56527d6591eefb28d028c25f","3f53ed340e9f722fac8fcc710cac60cc404873b9"],
#    ["1b3ccef1dabb7360d891f0460bb1f8a981aefca8","31933bba8ed3aab62d9dcc060d271e88522c73e6"],
#    ["6bfe7ea1397e70b4bea038b294ba6a61940354de","1f5314024e098934c01b7a0615b08014c84adb24"],
#    ["4d86c7a081b74595b925ffa64c9578e36b0f7374","284f156522dd1e00f07ee28f1fea7f400e2debec"],
#    ["2392872af4a958c41f3d817ae7c3bfe67d1a013a","d071eea3243834184b8dfb8fcc50ecf0aa29adfa"],
#    ["94bf679091944428c89919c52bb4cef8b2d92ca5","faaa55c05aae333d25e4bd26242e8229d17c3e2f"],
#    ["c5dd5836277c944f82072489e3f394053d603499","4fd7436b9b5fbecaa5155757738d976e208b4435"],
#    
#    ["","9524114075dd38ceb913a610a4b56da01d6d1bb4"]
#]



#   lucene-solr
#pathRepositoryFile = r"C:\Users\login\data\workspace\MLTool\datasets\lucene-solr\repositoryFile"
#pathRepositoryMethod = r"C:\Users\login\data\workspace\MLTool\datasets\lucene-solr\repositoryMethod"
#releasesIDs = [
#    ["","a0e7ee9d0d12370e8d2b5ae0a23b6e687e018d85"],
#    
#    None,
#    ["191558261298d2dcf24fdbd91b1d1727e69ea99c","b3f6ea5a814f05e9e8551b965e18567de814b62b"],
#    ["2e244caa1707bc82b7c487cc53ea9ebc4564c6fb","deecaac786bf460424c56eae72ac0d7f0305077d"],
#    ["4dc925bf4e198487ec455c5881d0c14030f8dd71","0fc2819690aa412322019e18c052bcff41e5d94d"],
#    ["429588097cdd0ec86dbf960d49bb1c0ac5d78b72","51b92ffe30c50bb08699200b62c40420378ac3df"],
#    ["cf7967cc467d9d697d520fcdf92fcdb52f7ddd4e","9bd00e9af390947b2751706503ba0d2f0b28da6d"],
#    ["a5402f68631768bae57d923613211128de077982","14162b1f8f2266547b5e1059f37c2efcf8981ea2"],
#    ["8c6e30536562413639eaf8bab1087da700733b33","ce6ac2f8144bf1a8fc35c555924357dd8efefc54"],
#
#    ["","2d062b9b9bc669298da80d215e455cf53377c8ce"]
#]



# opennms
#pathRepositoryFile = r"C:\Users\login\data\workspace\MLTool\datasets\opennms\repositoryFile"
#pathRepositoryMethod = r"C:\Users\login\data\workspace\MLTool\datasets\opennms\repositoryMethod"
#releasesIDs = [
#]


#   poi
#pathRepositoryFile =r"C:\Users\login\data\workspace\MLTool\datasets\poi\repositoryFile"
#pathRepositoryMethod = r"C:\Users\login\data\workspace\MLTool\datasets\poi\repositoryMethod"
#releasesFile = [
#    "b3f6ea5a814f05e9e8551b965e18567de814b62b",
#    None,
#    "b3f6ea5a814f05e9e8551b965e18567de814b62b",
#    "8ffd79b36803be29486ab6bdeff5dd02ea901927",
#    "99b9b7828cb268f62daa5b0c5639e549477b2c6f",
#    "382714eccd92667fc83f70115b736c64ebff9700"
#]
#releasesMethod = [
#    "6c234ab6c8d9b392bc93f28edf13136b3c053894",
#    None,
#    "0ac1311fd0a411f6cce60e48b595844c360af466",
#    "cc1cd7f005bc2379449439427d2c0acef7cd932a",
#    "37a74b4a8de772cde9788a6a8c2ae3c0862d31f3",
#    "449f08fc0003c64d09663715f28896a4dd010d6a"
#]


# realm-java
#pathRepositoryFile = r"C:\Users\login\data\workspace\MLTool\datasets\realm-java\repositoryFile"
#pathRepositoryMethod = r"C:\Users\login\data\workspace\MLTool\datasets\realm-java\repositoryMethod"
#releasesIDs = [
#    ["", "826e414d796147a7ad209fc378b004de994cf03b"],
#
#    ["eef492341a17351dc671b296bf7f1cd2c2ed32a4", "a55a4e0f9d296f79694f054a269d4d3e6ffcf4a9"],
#    ["1f19a05f820c1e43a9a0e38d8e32b0d96920df7f", "713b627781a904de3ef4651cb1b9c65436505dcc"],
#    ["66fb375b32b7db660c1d06fc7e27bf708d8cebab", "df3b168003f2f5f9e43f97a95b0cdefa5c77e5fc"],
#    ["e26255b9c5248620861565eb3caf7c908c4f8277", "4ef60a6e2f47f96b40580a5f115769315709512f"],
#    ["8740eb6ce5bfc8536be2b480988a6212f2ce8466", "7bdfcaa06c4dba11a5cbe6341b9142aafbdae79c"],
#    ["8a022573a5b095c2ec887720bd73098475829766", "2190de3870486baf6a5b7fd3d67135711470c1d6"],
#    ["5e1cb707bf37a4c5fa03b45cd5a4fde39fed146d", "363885a69f58225e99ec3f9c88670787bf5fd2da"],
#
#
#    ["", "626371b4e8379d066c8932795aaeb3ed3f4a539e"]
#]


# sonar-java
#pathRepositoryFile = r"C:\Users\login\data\workspace\MLTool\datasets\sonar-java\repositoryFile"
#pathRepositoryMethod = r"C:\Users\login\data\workspace\MLTool\datasets\sonar-java\repositoryMethod"
#releasesIDs = [
#    ["", "9140081865d82e930a0dd58db68c590880733be1"],
#
#    ["5ac4cf695248bc7385cb3377216cd86340bda0b0", "bdd64f010de25c0bdd08995a9fcc30f97c51adf2"],
#    ["fb4e10dcf17447ecea699934cdef0cf2009b1a52", "8112515981843759487f61adca77b49dadca85a9"],
#    ["65396a609ddface8b311a6a665aca92a7da694f1", "aa7fd785f6d8f34964b4bac744f69a86086ea848"],
#    ["b653c6c8640ab3d6015d036a060f58e027a653af", "a028375a33be6dedb0e6ea9d0a1caf23934a6155"],
#    ["fe9584c9f7812edc46f32d29440fc81b85a597a4", "bcd6639b73207c090f50178c577c0bca3081a23d"],
#    ["931433c0510b161974d4844679f7bf3c73bb3e37", "3922355ffa04b1fd531c1bad16e36a0e010bd12b"],
#
#    ["", "7e351f1d66566e809386516b97bdeab4c04b5b26"]
#
#]


#   wicket
pathRepositoryFile = r"C:\Users\login\data\workspace\MLTool\datasets\wicket\repositoryFile"
pathRepositoryMethod = r"C:\Users\login\data\workspace\MLTool\datasets\wicket\repositoryMethod"
releasesIDs = [
    None,

    None,
    None,
    None,
    None,
    None,
    ["0d75ee57abb31b4db48c0396870aba39c4d9ddee", "80bb2dbc0f0f8b4f89ba463f9886495f23f79e9b"],
    ["98a3a6295f426aa25a121f914a41bf792df8fdb0", "f4058e9842bb6197f139581df19be2057935973d"],
    ["5e789f1c98f6d57dba17f896c6220b0202af08a9", "25905159ad2ecd911a37d56422d22160880a1066"],
    ["c2802f3ef8df9833da63d144fb4ad03d59e31acc", "5350588b43f71bc2a8980d62e83d565626867582"],

    ["048e5681df960018722237ee6254273e839b5fd2", "2f1546f5f9ba8fc9fe71253675796f74fea5e953"]
]


def getCorrespondingCommit(commitIDMethod):#method->file
    print(commitIDMethod)
    for commit in RepositoryMining(pathRepositoryMethod).traverse_commits():
        if(commit.hash==commitIDMethod):
            commitDate = commit.committer_date
            print(commitDate)
    for commit in RepositoryMining(pathRepositoryFile).traverse_commits():
        if(commit.committer_date ==commitDate):
            return commit.hash
    return None



def identifyCommit_rbr(releaseIndex):
    if(releasesIDs[releaseIndex]==None):
        return None
    testRelease = releasesIDs[releaseIndex][1]
    if(releasesIDs[releaseIndex-1]==None):
        return None
    trainRelease = releasesIDs[releaseIndex-1][1]
    if(releasesIDs[releaseIndex-2]==None):
        return None
    previousTrainRelease = releasesIDs[releaseIndex-2][1]
    if(releaseIndex-2<0 or previousTrainRelease==None or trainRelease ==None):
        return None
    test  = ["R"+str(releaseIndex)+"_r_test"  ,  getCorrespondingCommit(testRelease),         trainRelease,  testRelease, releasesIDs[len(releasesIDs)-1][1]]
    train = ["R"+str(releaseIndex)+"_r_train" , getCorrespondingCommit(trainRelease), previousTrainRelease, trainRelease,                        testRelease]
    return train, test
def identifyCommit_cbc(commitHashes, releaseIndex, NumOfCommits):
    if(releasesIDs[releaseIndex]==None):
        return None
    testCommit = releasesIDs[releaseIndex][1]
    if(commitHashes.index(testCommit)-NumOfCommits*2<0):
        return None
    trainCommit = commitHashes[commitHashes.index(testCommit)-NumOfCommits]
    previousTrainCommit = commitHashes[commitHashes.index(testCommit)-NumOfCommits*2]
    test  = ["R"+str(releaseIndex)+"_c"+str(NumOfCommits) + "_test"  ,  getCorrespondingCommit(testCommit),         trainCommit,  testCommit, releasesIDs[len(releasesIDs)-1][1]]
    train = ["R"+str(releaseIndex)+"_c"+str(NumOfCommits) + "_train" , getCorrespondingCommit(trainCommit), previousTrainCommit, trainCommit, testCommit]
    return train, test
def identifyCommit_tbt(commits, releaseIndex, numOfMonths):
    if(1<=numOfMonths):
        if(releasesIDs[releaseIndex]==None):
            return None
        testCommit = releasesIDs[releaseIndex][1]
        sinceTrain = commits[commitHashes.index(testCommit)].committer_date - relativedelta(months = numOfMonths)
        untilTrain = commits[commitHashes.index(testCommit)].committer_date
        trainCommit = None
        for commit in reversed(commits):
            if(sinceTrain<commit.committer_date and commit.committer_date<untilTrain):
                trainCommit = commit.hash
        sincePreviousTrain = commits[commitHashes.index(testCommit)].committer_date - relativedelta(months = numOfMonths*2)
        previousTrainCommit = None
        for commit in reversed(commits):
            if(sincePreviousTrain<commit.committer_date and commit.committer_date < sinceTrain):
                previousTrainCommit = commit.hash
        if(previousTrainCommit ==None ):
            return None
        test  = ["R"+str(releaseIndex)+"_t"+str(numOfMonths) + "_test"  ,  getCorrespondingCommit(testCommit),             trainCommit,  testCommit, releasesIDs[len(releasesIDs)-1][1]]
        train = ["R"+str(releaseIndex)+"_t"+str(numOfMonths) + "_train" , getCorrespondingCommit(trainCommit),     previousTrainCommit, trainCommit, testCommit]
        return train, test
    #else:
    #    if(releasesIDs[releaseIndex]==None):
    #        return None
    #    testCommit = releasesIDs[releaseIndex][1]
    #    sinceTrain = commits[commitHashes.index(testCommit)].committer_date - relativedelta(days = 30*numOfMonths)
    #    untilTrain = commits[commitHashes.index(testCommit)].committer_date
    #    trainCommit = None
    #    for commit in reversed(commits):
    #        if(sinceTrain<commit.committer_date and commit.committer_date<=untilTrain):
    #            trainCommit = commit.hash
    #    sincePreviousTrain = commits[commitHashes.index(testCommit)].committer_date - relativedelta(days = 30 * numOfMonths * 2)
    #    previousTrainCommit = None
    #    for commit in reversed(commits):
    #        if(sincePreviousTrain<commit.committer_date and commit.committer_date < sinceTrain):
    #            previousTrainCommit = commit.hash
    #    if(previousTrainCommit ==None ):
    #        print("previousTrainCommit is none")
    #        return None
    #    test  = ["R"+str(releaseIndex)+"_t"+str(numOfMonths) + "_test"  ,  getCorrespondingCommit(testCommit),             trainCommit,  testCommit, releasesIDs[len(releasesIDs)-1][1]]
    #    train = ["R"+str(releaseIndex)+"_t"+str(numOfMonths) + "_train" , getCorrespondingCommit(trainCommit),     previousTrainCommit, trainCommit, testCommit]
    #    return train, test



records = []
commits = []
commitHashes = []
for commit in RepositoryMining(pathRepositoryMethod).traverse_commits():
    commitHashes.append(str(commit.hash))
    commits.append(commit)
for releaseIndex in range(len(releasesIDs)):
    print(releaseIndex)
    if(releaseIndex==0 or releaseIndex ==len(releasesIDs)-1):
        continue
    identifiedRecords = identifyCommit_rbr(releaseIndex)
    if(identifiedRecords!=None):
        records.extend(identifiedRecords)
    numOfCommitss = [500, 1000, 1500, 2000, 2500]
    for i, numOfCommits in enumerate(numOfCommitss):
        print(numOfCommits)
        identifiedRecords = identifyCommit_cbc(commitHashes, releaseIndex, numOfCommits)
        if(identifiedRecords!=None):
            records.extend(identifiedRecords)
    numOfMonthss = [1, 2, 3, 6, 12]
    for i, numOfMonths in enumerate(numOfMonthss):
        print(numOfMonths)
        identifiedRecords = identifyCommit_tbt(commits, releaseIndex, numOfMonths)
        if(identifiedRecords!=None):
            records.extend(identifiedRecords)

with open('commitIDs.csv', 'w') as f:
    writer = csv.writer(f, lineterminator="\n")
    writer.writerows(records)

#records = []
#commitsFile = []
#for commit in RepositoryMining(pathRepositoryFile).traverse_commits():
#    commitsFile.append(commit)
#commitsMethod = []
#for commit in RepositoryMining(pathRepositoryMethod).traverse_commits():
#    commitsMethod.append(commit)
#for idRelease in range(len(releasesFile)):
#    if(getPreviousRelease(idRelease)==None or releasesFile[idRelease]==None or  getFollowingRelease(idRelease)==None):
#        continue
#    recordRelease=[
#        ["R"+str(idRelease)+"_r"    , releasesFile[idRelease], None, releasesMethod[idRelease], releasesMethod[getFollowingRelease(idRelease)]],
#        ["R"+str(idRelease)+"_c500" , releasesFile[idRelease], None, releasesMethod[idRelease], releasesMethod[getFollowingRelease(idRelease)]],
#        ["R"+str(idRelease)+"_c1000", releasesFile[idRelease], None, releasesMethod[idRelease], releasesMethod[getFollowingRelease(idRelease)]],
#        ["R"+str(idRelease)+"_c1500", releasesFile[idRelease], None, releasesMethod[idRelease], releasesMethod[getFollowingRelease(idRelease)]],
#        ["R"+str(idRelease)+"_c2000", releasesFile[idRelease], None, releasesMethod[idRelease], releasesMethod[getFollowingRelease(idRelease)]],
#        ["R"+str(idRelease)+"_c2500", releasesFile[idRelease], None, releasesMethod[idRelease], releasesMethod[getFollowingRelease(idRelease)]],
#        ["R"+str(idRelease)+"_t1"   , releasesFile[idRelease], None, releasesMethod[idRelease], releasesMethod[getFollowingRelease(idRelease)]],
#        ["R"+str(idRelease)+"_t2"   , releasesFile[idRelease], None, releasesMethod[idRelease], releasesMethod[getFollowingRelease(idRelease)]],
#        ["R"+str(idRelease)+"_t3"   , releasesFile[idRelease], None, releasesMethod[idRelease], releasesMethod[getFollowingRelease(idRelease)]],
#        ["R"+str(idRelease)+"_t6"   , releasesFile[idRelease], None, releasesMethod[idRelease], releasesMethod[getFollowingRelease(idRelease)]],
#        ["R"+str(idRelease)+"_t12"  , releasesFile[idRelease], None, releasesMethod[idRelease], releasesMethod[getFollowingRelease(idRelease)]],
#        ["R"+str(idRelease)+"_t18"  , releasesFile[idRelease], None, releasesMethod[idRelease], releasesMethod[getFollowingRelease(idRelease)]],
#        ["R"+str(idRelease)+"_t24"  , releasesFile[idRelease], None, releasesMethod[idRelease], releasesMethod[getFollowingRelease(idRelease)]]
#    ]
#
#    recordRelease[0][1]=releasesFile[getPreviousRelease(idRelease)]
#    recordRelease[0][3]=releasesMethod[getPreviousRelease(idRelease)]
#
#    idReleaseCommit = releasesFile[idRelease]
#    print("idReleaseCommit: " + idReleaseCommit)
#
#    count = 0
#    for commit in commitsFile:
#        if(idReleaseCommit==commit.hash):
#            print(idReleaseCommit)
#            print(commit.hash)
#            break
#        count+=1
#    if(count-2500<0):
#        continue
#    recordRelease[1][1] = commitsFile[count-500].hash
#    print(commitsFile[count-500].hash)
#    recordRelease[2][1] = commitsFile[count-1000].hash
#    print(commitsFile[count-1000].hash)
#    recordRelease[3][1] = commitsFile[count-1500].hash
#    print(commitsFile[count-1500].hash)
#    recordRelease[4][1] = commitsFile[count-2000].hash
#    print(commitsFile[count-2000].hash)
#    recordRelease[5][1] = commitsFile[count-2500].hash
#    print(commitsFile[count-2500].hash)
#
#    test = None
#    bad = False
#    months = [1, 2, 3, 6, 12, 18, 24]
#    for i, month in enumerate(months):
#        begin = commitsFile[count].committer_date - relativedelta(months=month)
#        print("begin: " + str(begin))
#        end = commitsFile[count].committer_date
#        print("end: " + str(end))
#        for commit in reversed(commitsFile):
#            if(begin < commit.committer_date):
#                test = commit
#            else:
#                break
#        if(test==None):
#            bad = True
#            break
#        recordRelease[6+i][1] = test.hash
#        print(str(month) + "ヶ月最古: " + test.hash)
#        print(test.committer_date)
#        test = None
#    if(bad):
#        continue
#
#    idReleaseCommit = releasesMethod[idRelease]
#    print("idReleaseCommit: " + idReleaseCommit)
#    count = 0
#    for commit in commitsMethod:
#        if(idReleaseCommit==commit.hash):
#            break
#        count+=1
#    recordRelease[1][3] = commitsMethod[count-500].hash
#    print(commitsMethod[count-500].hash)
#    recordRelease[2][3] = commitsMethod[count-1000].hash
#    print(commitsMethod[count-1000].hash)
#    recordRelease[3][3] = commitsMethod[count-1500].hash
#    print(commitsMethod[count-1500].hash)
#    recordRelease[4][3] = commitsMethod[count-2000].hash
#    print(commitsMethod[count-2000].hash)
#    recordRelease[5][3] = commitsMethod[count-2500].hash
#    print(commitsMethod[count-2500].hash)
#
#    test = None
#    months = [1, 2, 3, 6, 12, 18, 24]
#    for i, month in enumerate(months):
#        begin = commitsMethod[count].committer_date - relativedelta(months=month)
#        print("begin: " + str(begin))
#        end = commitsMethod[count].committer_date
#        print("end: " + str(end))
#        for commit in reversed(commitsMethod):
#            if(begin < commit.committer_date):
#                test = commit
#            else:
#                break
#        recordRelease[6+i][3] = test.hash
#        print(str(month) + "ヶ月最古: " + test.hash)
#        print(test.committer_date)
#    records.extend(recordRelease)
#recordsCopy = copy.deepcopy(records)
#for record in recordsCopy:
#    recordCopy = copy.deepcopy(record)
#    recordCopy[0] = recordCopy[0]+"_test"
#    recordCopy[5] = releasesMethod[len(releasesMethod)-1]
#    records.append(recordCopy)
