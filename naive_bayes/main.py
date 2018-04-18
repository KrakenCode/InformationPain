#!/usr/bin/env python3
import naive_bayes
import pandas as pd
import sys


#Read input into rows and create data conversion tables
features = []
labels = []
df = pd.read_csv(sys.argv[1])
features = df.drop(['buys_computer'], axis=1).to_records(index=False)
labels = df['buys_computer']
print("features=")
print(features)
print("labels=")
print(labels)
'''with open(sys.argv[1]) as input_file:
    for line in input_file:
        fields = line.split(',')
        features.append(fields[:-1])
        labels.append(fields[-1])'''
clf = naive_bayes.NaiveBayesClassifier()
clf.fit(features, labels)
print(list(zip(features[-2:], clf.predict(features[-2:]))))
print('\n')
clf.print()
#print('labels[-20:]=%s' % [labels[-2:]])
