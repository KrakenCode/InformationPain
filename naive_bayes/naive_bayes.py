#!/usr/bin/env python3

#Naive Bayes Classifier
#Written by Luke Taylor

class NaiveBayesClassifier:
    def __init__(self):
        self.fitted = False
        self.p_given = []
        self.label_vals = {}
        self.p_classifier = {}
        pass

    def fit(self, features, labels):
        if(len(features) != len(labels)):
            raise IndexError("Feature and label lengths do not match")
        feature_vals = []
        for row in features:
            for k in range(len(row)):
                if(len(feature_vals) <= k):
                    feature_vals.append({})
                feature_vals[k][row[k]] = True
        self.label_vals = {}
        for label in labels:
            self.label_vals[label] = True

        total = len(features)

        #Find proportions and counts for classifier attribute
        class_counts = {}
        for val in self.label_vals:
            class_counts[val] = len(list(filter(lambda x: x == val, labels)))
            self.p_classifier[val] = class_counts[val]/total

        print('p_classifier=%s' % str(self.p_classifier))
        print('feature_vals=%s' % str(feature_vals))
        #Find all probabilites for attributes
        self.p_given = []
        for k in range(len(feature_vals)):
            self.p_given.append({})
            for val in feature_vals[k]:
                self.p_given[k][val] = {}
                for label in self.label_vals:
                    self.p_given[k][val][label] = 1
                    for j in range(len(features)):
                        if(features[j][k] == val and labels[j] == label):
                            self.p_given[k][val][label] += 1
                    self.p_given[k][val][label] /= class_counts[label]
        #print('p_given:\n%s' % json.dumps(self.p_given, indent=4))
        self.fitted = True

    #Predict classifier attribute and print results
    def predict(self, features):
        if(not self.fitted):
            raise Exception('Need to fit data before predicting')
        #print('features=%s' % json.dumps(features, indent=2))
        predicted_labels = []
        for row in features:
            prod = {}
            for label in self.label_vals:
                prod[label] = 1.0
                for k in range(len(row)):
                    prod[label] *= self.p_given[k][row[k]][label]
                prod[label] *= self.p_classifier[label]
            desc = ','.join(row)
            max = 0.0
            max_label = ''
            for label in self.label_vals:
                if(prod[label] > max):
                    max = prod[label]
                    max_label = label
            print('class(%s)=%s' % (desc, max_label))
            predicted_labels.append(max_label)
        return predicted_labels