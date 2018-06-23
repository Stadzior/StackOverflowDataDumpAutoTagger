import re
import operator
from os import listdir
from os.path import isfile, join
from textblob.classifiers import NaiveBayesClassifier

trainPart = 0.05  
testPartCount = 5
tagsMatchCount = 3

if (trainPart < 0 or trainPart > 100):
    exit
posts = open("Posts.xml", "r").read().decode()    

#Building training set
trainSetCount = int(len(posts)*trainPart)
print("Analyzing: " + str(trainSetCount) + " posts.")
trainPosts = posts[:trainSetCount]
trainSet = [(re.sub(re.compile('<.*?>'), '', post.get("Body")).replace("\n", ""), tag) for tag in post.get("Tags").replace("<", "").split(">")[0:len(tags)-1] for post in list(filter(lambda post: post.get("Tags") is not None, trainPosts))]

#trains classifier
classifier = NaiveBayesClassifier(trainSet)

#Build test set
testSet = posts[trainSetCount:trainSetCount+testPartCount]

#Perform classification
for post in testSet:
    classificationResults = classifier.prob_classify(re.sub(re.compile('<.*?>'), '', post.get("Body")).replace("\n", ""))
    print("\nPost: " + classificationResults)
    print("Best matching tags: ")
    tagsWithRanks = {}
    for tag in classificationResults.samples():
        tagsWithRanks[tag] = classificationResults.prob(tag)

    tagsWithRanks = sorted(tagsWithRanks.items(), key=operator.itemgetter(1), reverse=True)
    for tag in tagsWithRanks[0:tagsMatchCount]:
        print(tag[0] + ", " + str(tag[1]*100))