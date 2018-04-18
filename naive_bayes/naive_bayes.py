#!/usr/bin/env python3

#Naive Bayes Classifier
#Written by Luke Taylor

class NaiveBayesClassifier:
    def __init__(self):
        self.fitted = False
        self.p_given = []
        self.label_vals = {}
        self.p_classifier = {}
        self.numerical_rows = {}
        self.types = []
        pass

    def __factorial(self, n):
        ans = 1
        for k in range(2,n+1):
            ans *= k
        return ans

    def __erf(self, z):
        total = 0.0
        neg = 1
        for n in range(30):
            total += neg*z**(2*n+1)/(self.__factorial(n)*(2*n+1))
            neg *= -1
        return total*2.0/(3.141592653589**0.5)

    def __get_mu_delta(self, arr):
        mean = 0.0
        for k in arr:
            mean += k
        mean /= len(arr)
        stddev = 0.0
        for k in arr:
            stddev += (k-mean)**2
        stddev /= len(arr)-1
        stddev = stddev**0.5
        return (mean, stddev)

    def __gaussian_func_generator(self, mu, delta):
        def prob(x):
            return 0.5*(1+self.__erf((x-mu)/(delta*(2**0.5))))
        return prob
    
    def fit(self, features, labels):
        #print(self.__factorial(5))
        #print(self.__erf(2.5))
        #print(self.__gaussian_func_generator(0,1)(1))
        if(len(features) != len(labels)):
            raise IndexError("Feature and label lengths do not match")
        feature_vals = []
        self.types = []
        for row in features:
            for k in range(len(row)):
                if(len(self.types) <= k):
                    if(str(row[k]) == row[k]):
                        self.types.append('nominal')
                    else:
                        self.types.append('numeric')
                if(len(feature_vals) <= k):
                        feature_vals.append({})
                if(self.types[k] == 'nominal'):
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

        #print('p_classifier=%s' % str(self.p_classifier))
        #print('feature_vals=%s' % str(feature_vals))
        #Find all probabilites for attributes
        self.p_given = []
        for k in range(len(feature_vals)):
            if(self.types[k] == 'numeric'):
                self.p_given.append({})
                for label in self.label_vals:
                    vals_in_class = []
                    for j in range(len(features)):
                        if(labels[j] == label):
                            vals_in_class.append(features[j][k])
                    mean,stddev = self.__get_mu_delta(vals_in_class)
                    self.p_given[k][label] = self.__gaussian_func_generator(mean, stddev)
            else:
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
        #print("self.types=" + str(self.types))
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
                    if(self.types[k] == 'numeric'):
                        prod[label] *= self.p_given[k][label](row[k])
                    else:
                        prod[label] *= self.p_given[k][row[k]][label]
                prod[label] *= self.p_classifier[label]
            desc = ','.join(map(str,row))
            #print(prod)
            max = 0.0
            max_label = ''
            for label in self.label_vals:
                if(prod[label] > max):
                    max = prod[label]
                    max_label = label
            #print('class(%s)=%s' % (desc, max_label))
            predicted_labels.append(max_label)
        return predicted_labels

    def print(self):
        print(self.p_given)