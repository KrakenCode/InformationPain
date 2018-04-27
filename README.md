# InformationPain
Data Mining group project


**Requirements**
```
python              --  3.6
jupyter notebook    --  5.41
scipy               --  1.0.0
sklearn             --  0.19.1
pandas              --  0.22.0
numpy               --  1.14.2
tqdm                --  4.23.0
```

To use our models, it's as simple as importing and instantiating the class, call fit with the feature and their labels, and then call predict with the test features.

**ID3**
```
from ID3 import ID3
id3 = ID3()
id3.fit(features_train, labels_train)
predictions = id3.predict(features_test)
```

**Naive Bayes**
```
from Naive_Bayes import NaiveBayesClassifier
nbc = NaiveBayesClassifier()
nbc.fit(features_train, labels_train)
nbc_prediction = nbc.predict(features_test)
print(classification_report(labels_test, nbc_prediction))
```

**Neural Network**
- The training labels must be one hot encodings, even if they are numeric (in the future the network can do the conversion for the user)
- If the features are nominal then they will have to be converted into one hot encodings as well
- When instantiating the network you have to give the correct shape for the input and output layers.
- Any integer between the first and last element of the layer_structures corresponds to the amount of neurons in the relative hidden layer
- The network defaults to sigmoid for its activation function, but relu can be imported and used.
- If you need to retrain the network, create a new object; as of now the weights don't reset

```
from Neural_Network import Network, relu
# convert labels to one hots for the neural network
labels_train_one_hot = pd.get_dummies(labels_train).as_matrix()

neuralnet = Network(layer_structures=[40, 10, 10, 3], activation_function=relu)
neuralnet.fit(features_train, labels_train_one_hot, iterations=5000)
neuralnet_prediction = neuralnet.predict(features_test, classification=True)
```


**Putting it all together example:**
```
features, labels = make_classification(n_samples=3000, n_classes=3, n_informative=20,
                                       			n_features=30, class_sep=1.5)
kf = KFold(n_splits=10, shuffle=True)

id3 = ID3()
naive_bayes = NaiveBayesClassifier()

labels_true = []
id3_predicted = []
naive_bayes_predicted = []
neuralnet_predicted = []

for train_index, test_index in kf.split(features):
    features_train, features_test = features[train_index], features[test_index]
    labels_train, labels_test = labels[train_index], labels[test_index]

    # save all the of test labels
    labels_true.extend(labels_test)

    # convert labels to one hots for the neural network
    # features are numeric, so they don't have to be converted
    labels_train_one_hot = pd.get_dummies(labels_train).as_matrix()

    # recreate Neural Network, for now it doesn't reset its weight, thus make a new one
    # 30 feature neurons, 2 hidden layers with 15 neurons each, 3 output neurons
    neuralnet = Network(layer_structures=[30, 15, 15, 3], activation_function=relu)

    # train models
    id3.fit(features_train, labels_train)
    naive_bayes.fit(features_train, labels_train)
    neuralnet.fit(features_train, labels_train_one_hot, iterations=5000)

    # get predictions
    id3_predicted.extend(id3.predict(features_test))
    naive_bayes_predicted.extend(naive_bayes.predict(features_test))
    neuralnet_predicted.extend(neuralnet.predict(features_test, classification=True))


print('\n\nID3 Decision Tree:')
print('Accuracy: {:.2f}'.format(accuracy_score(labels_true, id3_predicted)))
print(classification_report(labels_true, id3_predicted))

print('\n\nNaive Bayes:')
print('Accuracy: {:.2f}'.format(accuracy_score(labels_true, naive_bayes_predicted)))
print(classification_report(labels_true, naive_bayes_predicted))

print('\n\nOur Neural Network:')
print('Accuracy: {:.2f}'.format(accuracy_score(labels_true, neuralnet_predicted)))
print(classification_report(labels_true, neuralnet_predicted))
```
