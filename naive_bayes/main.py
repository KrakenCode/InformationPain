import naive_bayes
import sys


#Read input into rows and create data conversion tables
features = []
labels = []
with open(sys.argv[1]) as input_file:
    for line in input_file:
        fields = line.split(',')
        features.append(fields[:-1])
        labels.append(fields[-1])
clf = naive_bayes.NaiveBayesClassifier()
clf.fit(features[:-1], labels[:-1])
clf.predict(features[-1:])
#print('labels[-20:]=%s' % [labels[-2:]])
