import numpy as np
from sklearn import svm
import os
from sklearn.model_selection import train_test_split

class config:

    filepath = "../../Downloads/pac_data/"
    bad_files = [".DS_Store"]

#passes in a local file related to the indexed directories
#that will then be loaded
def loadDatapath(filename = None):
    def loadExamples(currFilename):
        print 'curr_file_name: ', currFilename
        #nextFiles = ['0', '1']
        nextFiles = os.listdir(currFilename)
        print 'nf: ', nextFiles
        currExamples = []
        for nextFile in nextFiles:
            if nextFile in config.bad_files: continue
            currPath = currFilename + nextFile
            #print os.listdir(currFilename + nextFile)
            image_list = os.listdir(currPath)
            print 'image_list_len: ', len(image_list)
            image_limit = 0
            for imageFile in image_list:
                if image_limit > 10:
                    break
                image_limit += 1
                if imageFile in config.bad_files:
                    print 'bad_FILE'
                    continue
#                print imageFile
#                with open(currPath + "/" + imageFile, "rb") as f:
                currImage = np.load(currPath + "/" + imageFile)['x']
                #everything here is a matrix
                #print 'currImage shape: ', currImage.shape
                currImage = currImage.flatten()
                currExamples.append((currImage, nextFile))
                #print 'currExamples_len: ', len(currExamples)
                #print currImage
                #break
        return currExamples
    assert(filename is not None)
#   print os.listdir(config.filepath + filename)

    currExamples = loadExamples(config.filepath + filename + "/")
    return currExamples

#I'm going to be honest, I don't know if this code works
#It only partially works, and it'll break in some edge cases
def importData():
#    print os.listdir(config.filepath)
    dataset = np.array([])
    for i, filename in enumerate(os.listdir(config.filepath)):
        print 'i: ', i
        if filename in config.bad_files: continue
        currDataset = loadDatapath(filename)
        if i == 1:
            dataset = currDataset
        else:
            print 'importData-dataset len: ', len(dataset), 'loadDatapath(filename_len)', len(currDataset)
            dataset = np.concatenate((dataset, loadDatapath(filename)))
        #break
    return dataset


def runMachineLearning(trainingSet):

    X, y = np.array([currVal[0] for currVal in trainingSet]), [currVal[1] for currVal in trainingSet]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    #print  data, 'shape: ', data.shape
    clf = svm.SVC()
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    num_correct = 0
    for y,p in enumerte(predictions):
        if p == y:
            num_correct += 1

    percent_correct = num_correct / (len(predictions) * 1.0)

    print 'predictions: ', predictions, 'labels: ', labels
    print 'percent correct: ', percent_correct




def main():
    dataset = importData()
    print 'dataset: ', dataset, 'dataset len: ', len(dataset)
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    runMachineLearning(dataset)

if __name__ == "__main__":
    main()
