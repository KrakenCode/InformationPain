import pandas as pd
import numpy as np
from pandas import DataFrame
from scipy.stats import mode


class ID3:
    # The node for the ID3 decision tree
    class TreeNode:
        def __init__(self, attrId = -1, matchesLevel = [], children = [], returnVal = None):
            # The index of the column to split out
            self.attrId = attrId

            # List of one parameter lambda functions that return True/False
            # based on whether the parameter matches the branch
            self.matchesLevel = matchesLevel

            # List of child TreeNodes
            self.children = children

            # The return value for this node, the class label if it's a leaf node, None if it is not
            self.returnVal = returnVal

    def __init__(self):
        self.rootNode = None

    # Determines the best attribute to split by given features X and labels y
    # Currently, this function assumes that X is numeric, and therefore computes
    # split points by taking the average of each pair of ordered unique values
    # Returns the column index of the best attribute and a list of functions
    # that return True/False based on whether the attribute fits the split
    def determineBestAttribute(X, y):

        # Computes the information in the labels
        def computeInfo(y):
            yVals = np.unique(y)
            infoComps = np.zeros(len(yVals))
            total = len(y)
            for i in range(len(yVals)):
                count = len(y[y == yVals[i]])
                component = -(count / total) * np.log2(count / total)
                infoComps[i] = component
            return infoComps.sum()

        # array of information gain/other metric for each column, initialized to zeros
        metric = np.zeros(X.shape[1])

        # list of list of functions that return True/False based on level of attribute, initially empty
        matchFuncs = []

        # Compute best information and match functions for each column
        for i in range(X.shape[1]):

            # check if attibute is categorical
            if not np.issubdtype(X.dtype, np.number):
                # Find all unique values and create functions to split on each value
                vals = np.unique(X[:, i])
                splitFuncs = [lambda x, v=v: x == v for v in vals]

                # Split the labels by each function
                ySplits = [y[f(X[:, i])] for f in splitFuncs]

                # Compute the information gain for each split
                infos = [computeInfo(ySplit) for ySplit in ySplits]

                # Take a weighted sum to get the total metric for this column
                metric[i] = np.dot([len(split) for split in ySplits], infos) / len(y)
                matchFuncs.append(splitFuncs)
                continue

            # create split points for the column
            uniqueVals = np.unique(X[:, i])
            splitPoints = (uniqueVals[:-1] + uniqueVals[1:]) / 2

            # Iterate through the split points, find the best one for this specific column
            bestMetric = None
            splitFuncs = None

            for sp in splitPoints:
                matchFunc1 = lambda x, sp=sp: x < sp
                matchFunc2 = lambda x, sp=sp: x >= sp
                y1 = y[matchFunc1(X[:, i])]
                y2 = y[matchFunc2(X[:, i])]
                info1 = computeInfo(y1)
                info2 = computeInfo(y2)
                m = len(y1) / len(y) * info1 + len(y2) / len(y) * info2
                if bestMetric == None or m < bestMetric:
                    splitFuncs = [matchFunc1, matchFunc2]
                    bestMetric = m

            # save the best metric (ie information gain) for the column and the match function for it
            metric[i] = bestMetric
            matchFuncs.append(splitFuncs)

        # Find the best attribute/match function from all the columns
        bestAttr = np.argmin(metric)
        return bestAttr, matchFuncs[bestAttr]

    # Creates a node in the tree with feature set X and labels y
    def createNode(X, y):
        # If there is only 1 unique label, predict that label
        if len(np.unique(y)) == 1:
            return ID3.TreeNode(returnVal = y[0])

        # No features left, return the most abundant class in y
        if X.shape[1] == 0:
            return ID3.TreeNode(returnVal = mode(y).mode[0])

        # Determine the best attribute to split the dataset by and how to split it
        bestAttr, matchesLevel = ID3.determineBestAttribute(X, y)

        # Create the children nodes with subsets of the features/labels
        children = []
        for i in range(len(matchesLevel)):
            children.append(
                ID3.createNode(
                    X[matchesLevel[i](X[:, bestAttr]), :], y[matchesLevel[i](X[:, bestAttr])]
                )
            )
        return ID3.TreeNode(attrId = bestAttr, matchesLevel = matchesLevel, children = children)

    # Fits the ID3 decision tree with feature set X and labels y
    def fit(self, X, y):
        self.rootNode = ID3.createNode(X, y)

    def predict(self, X):
        if self.rootNode == None:
            return None
        # Given a row from the feature set, predict the label
        def predictTuple(x, node = self.rootNode):
            if node.returnVal != None:
                return node.returnVal
            for i in range(len(node.matchesLevel)):
                f = node.matchesLevel[i]
                if f(x[node.attrId]):
                    return predictTuple(x, node.children[i])
            # Attribute has a new value not seen before by the tree, returning None for the label
            return None

        # Iterate through all the rows, predict the label for each
        preds = [None for v in range(X.shape[0])]
        for i in range(X.shape[0]):
            preds[i] = predictTuple(X[i, :])
        return preds

    # For debugging
    def printTreeBreadth(self):
        strStack = []
        nodeStack = [(0, self.rootNode)]
        while len(nodeStack) != 0:
            level, n = nodeStack.pop(0)
            for i in range(level):
                print("  ", end="")
            if n.returnVal != None:
                print("Return val of", n.returnVal)
                continue
            print("Attribute", n.attrId)
            for c in n.children:
                nodeStack.append((level + 1, c))

    def printTreeDepth(self):
        def printNode(n = self.rootNode, level = 0):
            for i in range(level):
                print("  ", end = "")
            if n.returnVal != None:
                print("Return val of", n.returnVal)
                return
            print("Attribute", n.attrId)
            for c in n.children:
                printNode(c, level + 1)
        printNode()
