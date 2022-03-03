# A random forest algorithm to predict heart disease. 
# Based on the 'Heart Failure Prediction Dataset' taken from Kaggle 
# (https://www.kaggle.com/fedesoriano/heart-failure-prediction).

# Created by Kento Seki between 10th December and 22nd December 2021.
# Updated between 4th Jan 2022 and 20th Feb 2022. 


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# TO DO:
# - Fix bug: set seed 4807, forestSizes list(range(50)) and attrSubSizes list(range(2,6)) 
# ----> some rows of the tallies do NOT add up to the forestSize

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


# THINGS TO THINK ABOUT: 
 # - maximum tree depth as hyperparameter?
# - more evaluation methods than just accuracy...
#   https://machinelearningmastery.com/classification-accuracy-is-not-enough-more-performance-measures-you-can-use/ 

# Evaluate model using a validation set allocated at the beginning
# --> no longer using out-of-bag error as the evaluation measure

from classes import Node, Tree
import pandas as pd
import random

# Finds the best random forest model out of every combination of numAttrs (from
# attrSubSizes) and numTrees (from forestSizes). Uses the entire training data
# (train) passed into the function (currently without bootstrapping). 
# Predictions on the valudation dataset are made simultaneously by splitting the
# validation data through the tree along with the training data. Returns the 
# model (stored as list of decision trees) with the highest accuracy on val, 
# along with the number of indecisive predictions made and the model's
# hyperparameters.
def findBestForest(train, val, attrSubSizes, forestSizes):

    best = {'forest': None, 'numTrees': None, 'numAttrs': None,
            'valAcc': 0, 'valnUnsure': None}

    # Iterate through attrSubSizes in descending order to make the algorithm
    # speed up towards the end, rather than slow down.
    attrSubSizes.sort(reverse=True)

    for numAttrs in attrSubSizes:

        forest = []
        # Tally of validation set predictions
        emptyTally = {'Disease': [0] * len(val), 'Healthy': [0] * len(val)}
        tally = pd.DataFrame(emptyTally)
        
        for numTrees in forestSizes:
            
            # Add trees to model until we reach the next desired size
            while len(forest) != numTrees:
                newTree = Tree()
                newTree.root.data = train
                newTree.root.unusedAttrs = list(range(11))
                forest.append(newTree)
                # Build the decision tree by recursively splitting nodes
                nodeSplit(newTree.root, numAttrs, val, tally)
                # ^ Splits the validation data WHILE building the tree and 
                #   tallies predictions every time a leaf node is reached

            result = getAccuracy(tally, val)
            acc, unsure = result[0], result[1]
            if acc > best['valAcc']:
                pd.set_option("display.max_rows", None, "display.max_columns", None) ###
                print(tally[tally['Healthy'] + tally['Disease'] != numTrees]) ###
                best['forest'] = forest.copy()
                best['numTrees'] = numTrees
                best['numAttrs'] = numAttrs
                best['valAcc'] = acc
                best['valnUnsure'] = unsure
                print(f"* Improved model! | {numTrees} trees, {numAttrs} "\
                    f"attributes | Validation accuracy: {acc}% | "\
                    f"Indecisive predictions: {unsure}")

            # Update % progress
            done = attrSubSizes.index(numAttrs)*len(forestSizes)\
                + forestSizes.index(numTrees) + 1
            total = len(attrSubSizes) * len(forestSizes)
            perc = round(done/total * 100, 2)
            print(f'Progress: {perc}%', end="\r")

    return best

################################################################################

# Calculates the accuracy of the model's predictions on the validation set from
# the tally.
def getAccuracy(tally, val):
    # Use tally to calculate acc (accuracy) and nUnsure
    nCorrect, nUnsure = 0, 0
    for i in range(len(val)):
        if tally.iloc[i,0] > tally.iloc[i,1] and val.iloc[i, 11] == 1:
            nCorrect += 1
        elif tally.iloc[i,0] < tally.iloc[i,1] and val.iloc[i, 11] == 0:
            nCorrect += 1
        elif tally.iloc[i,0] == tally.iloc[i,1]:
            nUnsure += 1

    acc = round(nCorrect/len(val) * 100, 2)

    return acc, nUnsure

################################################################################

# Change categorical variables to numerical
def catToNum(df):
    # Sex
    df = df.replace(to_replace = "M", value = 0)
    df = df.replace(to_replace = "F", value = 1)
    # ChestPainType
    df = df.replace(to_replace = "TA", value = 0)
    df = df.replace(to_replace = "ATA", value = 1)
    df = df.replace(to_replace = "NAP", value = 2)
    df = df.replace(to_replace = "ASY", value = 3)
    # RestingECG
    df = df.replace(to_replace = "Normal", value = 0)
    df = df.replace(to_replace = "ST", value = 1)
    df = df.replace(to_replace = "LVH", value = 2)
    # Exercise Angina
    df = df.replace(to_replace = "Y", value = 0)
    df = df.replace(to_replace = "N", value = 1)
    # ST_Slope
    df = df.replace(to_replace = "Up", value = 0)
    df = df.replace(to_replace = "Flat", value = 1)
    df = df.replace(to_replace = "Down", value = 2)
    return df

################################################################################

# Recursively split the data at the given node (in preorder traversal order: 
# root->left->right). Also split the validation data (val) based on decisions
# made using the training data.
# In the base case (node is a leaf node), add to the True/False predictions
# tally for each row in the validation data at that node.
def nodeSplit(node, numAttrs, val, tally):
    df = node.data
    # Base case for not splitting
    if len(df) <= 5 or len(node.unusedAttrs) == 0:
        node = assignLabel(node)
        tallyValPredictions(node, val, tally)
        return None

    # Calculate impurity of the node as is
    currImp = 1 - (len(df[df['HeartDisease'] == 1]) / len(df))**2 \
                - (len(df[df['HeartDisease'] == 0]) / len(df))**2

    # Obtain a random sample of attributes (numAttrs determines how many)
    # HOWEVER, if unusedAttrs < numAttrs, then just take all unusedAttrs
    if len(node.unusedAttrs) > numAttrs:
        candidateAttrs = random.sample(node.unusedAttrs, k = numAttrs)
    else:
        candidateAttrs = node.unusedAttrs

    # Try splitting by each attribute. Record the lowest impurity and the 
    # attribute + splitpoint that obtains it
    minImp, minPt, minAttr = 1, -1, -1
    for attr in candidateAttrs:
        attrName = df.columns[attr]
        spltPts = findSplitPoints(attrName, df)
        # Calculate impurity for each splitpoint and store if it's the min
        for pt in spltPts:
            imp = splitImp(attrName, pt, df)
            if imp < minImp:
                minImp, minPt, minAttr = imp, pt, attr
        
        # Check pairwise splits/choices too, if the attribute is ChestPainType
        if attrName == 'ChestPainType':
            cInfo = PairwiseChestSplits(df)
            if cInfo[0] < minImp:
                minAttr, minImp, minPt = attr, cInfo[0], cInfo[1]

    # Make the best split found (best outcome could be no split)
    if currImp <= minImp:
        node = assignLabel(node)
        tallyValPredictions(node, val, tally)
        return None
    elif df.columns[minAttr] == 'ChestPainType' and isinstance(minPt, list):
        # ...this means we did a pairwise split of ChestPainType
        node.splitAttr = minAttr
        node.splitPt = minPt
        newUnusedAttrs = node.unusedAttrs
        newUnusedAttrs.remove(minAttr)
        
        # Split data between two new children
        node.left, node.right = Node(), Node()
        node.left.data = df[(df['ChestPainType'] == minPt[0])\
            | (df['ChestPainType'] == minPt[1])]
        node.left.unusedAttrs = newUnusedAttrs
        node.right.data = df[(df['ChestPainType'] != minPt[0])\
            & (df['ChestPainType'] != minPt[1])]
        node.right.unusedAttrs = newUnusedAttrs

        # Split validation data between new children
        valLeft = val[(val['ChestPainType'] == minPt[0])\
            | (val['ChestPainType'] == minPt[1])]
        valRight = val[(val['ChestPainType'] != minPt[0])\
            & (val['ChestPainType'] != minPt[1])]
    else:
        node.splitAttr = minAttr
        node.splitPt = minPt
        newUnusedAttrs = node.unusedAttrs
        newUnusedAttrs.remove(minAttr)
        attrName = df.columns[minAttr]
        
        # Split data between two new children
        node.left, node.right = Node(), Node()
        node.left.data = df[df[attrName] < minPt]
        node.left.unusedAttrs = newUnusedAttrs
        node.right.data = df[df[attrName] > minPt]
        node.right.unusedAttrs = newUnusedAttrs

        # Split validation data between new children
        valLeft = val[val[attrName] < minPt]
        valRight = val[val[attrName] > minPt]

    nodeSplit(node.left, numAttrs, valLeft, tally)
    nodeSplit(node.right, numAttrs, valRight, tally)

################################################################################

# Calculate the impurity resulting from a split by the given attribute (attr), 
# at the given point (pt), on the data (df) at the node.
def splitImp(attrName, pt, df):
    # rows with attr < pt (left subtree)
    # (legend: lfTrue = left subtree and has heart dis)
    lfTrue = df[(df[attrName] < pt) & (df['HeartDisease'] == 1)]
    lfFalse = df[(df[attrName] < pt) & (df['HeartDisease'] == 0)]
    numLf = len(lfTrue) + len(lfFalse)
    # rows with attr > pt (right subtree)
    rtTrue = df[(df[attrName] > pt) & (df['HeartDisease'] == 1)]
    rtFalse = df[(df[attrName] > pt) & (df['HeartDisease'] == 0)]
    numRt = len(rtTrue) + len(rtFalse)

    # Calculate Gini impurity
    lfImp = 1 - ( len(lfTrue) / numLf )**2   \
                - ( len(lfFalse) / numLf )**2
    rtImp = 1 - ( len(rtTrue) / numRt )**2   \
                - ( len(rtFalse) / numRt )**2
    # imp = weighted average of leftImp and rightImp (weighted by 
    # number of rows in the left and right subtrees)
    imp = lfImp * ( numLf / len(df) ) + rtImp * ( numRt / len(df) )
    return imp

################################################################################

# Finds the points BETWEEN the distinct values of attrName, each of which we 
# should try splitting the data by.
def findSplitPoints(attrName, df):
    values = df[attrName].unique()
    values.sort()
    spltPts = []
    for i in range(len(values) - 1):
        spltPts.append((values[i] + values[i+1]) / 2)
    return spltPts

################################################################################

# Tries splitting the data by pairs of ChestPainType that CANNOT be achieved
# with a simple splitPoint bewteen 0,1,2,3 (e.g. sending rows with 0 and 2 to 
# one child while sending the rest to the other child, cannot be achieved with 
# a splitPoint).
def PairwiseChestSplits(df):
    # Store the minimum attainable impurity in info[0]
    # Store a list with the two ChestPainType values sent to one node in info[1]
    # -- Possible pair choices: 0-2, 0-3, 1-2, 1-3
    info = [1, []]
    for i in [0,1]:
        for j in [2,3]:
            # Send rows with ChestPainType i or j to one side, and all other 
            # rows to the other side - calculate resulting impurity
            left = df[(df['ChestPainType'] == i) | (df['ChestPainType'] == j) ]
            numLf = len(left)
            lfTrue = len(left[left['HeartDisease'] == 1])
            lfFalse = len(left[left['HeartDisease'] == 0])

            right = df[(df['ChestPainType'] != i) & (df['ChestPainType'] != j)]
            numRt = len(right)
            rtTrue = len(right[right['HeartDisease'] == 1])
            rtFalse = len(right[right['HeartDisease'] == 0])
            
            if numLf == 0 or numRt == 0:
                continue # <--------------------------------------------- (22.2.22) What was this if statement for?
            lfImp = 1 - ( lfTrue / numLf )**2   \
                        - ( lfFalse / numLf )**2
            rtImp = 1 - ( rtTrue / numRt )**2   \
                        - ( rtFalse / numRt )**2
            imp = lfImp * ( numLf / len(df) ) + rtImp * ( numRt / len(df) )

            if imp < info[0]:
                info[0] = imp
                info[1] = [i,j]
    
    return info

################################################################################

# Assign label to a leaf node based on which categorisation is most common in
# its data rows (labelled either Healthy or Disease - this is our answer for
# any data row that reaches this node when using the tree)
def assignLabel(node):
    df = node.data
    if all(df.mode()['HeartDisease'].dropna()) == 0:
        #print('False is the most common in this leaf node!')
        node.label = 'Healthy'
    elif all(df.mode()['HeartDisease'].dropna()) == 1:
        #print('True is the most common in this leaf node!')
        node.label = 'Disease'
    else:
        print('something is very wrong')
    return node

################################################################################

# Adds to the tally of predictions for the validation dataset for all the 
# validation data rows that arrived at the given leaf node.
def tallyValPredictions(node, val, tally):
    for i in range(len(val)):
        row = val.iloc[i].name
        if node.label == 'Healthy':
            tally.iloc[row, 1] += 1
        elif node.label == 'Disease':
            tally.iloc[row, 0] += 1
        else:
            print("******** NODE IS UNLABELED!!! ********")

################################################################################

# Passes the testing dataset (df) through each tree in the random forest and 
# tallies their predictions. Compares the tallied predictions to the actual
# known results of the testing dataset to determine the percentage accuracy.
# Returns the accuracy.
def testModel(forest, df):
    emptyTally = {'Disease': [0] * len(df), 'Healthy': [0] * len(df)}
    tally = pd.DataFrame(emptyTally)

    for tree in forest:
        passData(tree.root, df, tally)

    nCorrect, nUnsure = 0, 0
    for i in range(len(tally)):
        if df.iloc[i, 11] == 0 and tally.iloc[i,0] < tally.iloc[i,1]:
            nCorrect += 1
        elif df.iloc[i, 11] == 1 and tally.iloc[i,0] > tally.iloc[i,1]:
            nCorrect += 1
        elif tally.iloc[i,0] == tally.iloc[i,1]:
            nUnsure += 1

    acc = round(nCorrect/len(df) * 100, 2)
    return acc, nUnsure

################################################################################

# Recursively passes the testing dataset (df) through each node in the decision
# tree. Once a leaf node is encountered, the label of that node is used to 
# add to the tally of predictions
def passData(node, df, tally):
    # Base case - leaf node
    if node.left is None and node.right is None:
        # Add to tally using labels
        for i in range(len(df)):
            rowIndex = df.iloc[i].name
            if node.label == 'Healthy':
                tally.iloc[rowIndex, 1] += 1
            elif node.label == 'Disease':
                tally.iloc[rowIndex, 0] += 1
        return None

    # Recursive case - split data between children nodes
    minAttr, minPt = node.splitAttr, node.splitPt
    attrName = df.columns[minAttr]
    if attrName == 'ChestPainType' and isinstance(minPt, list):
        # ...this means the node has a pairwise split of ChestPainType
        dfLeft = df[(df['ChestPainType'] == minPt[0])\
            | (df['ChestPainType'] == minPt[1])]
        dfRight = df[(df['ChestPainType'] != minPt[0])\
            & (df['ChestPainType'] != minPt[1])]
    else:
        dfLeft = df[df[attrName] < minPt]
        dfRight = df[df[attrName] > minPt]

    passData(node.left, dfLeft, tally)
    passData(node.right, dfRight, tally)

################################################################################

# Displays hyperparameter and performance information about the given model.
def modelInfo(train, val, hold, model):
    # Calculate no-info rates for comparison
    valNoInfo = noInfoRate(val)
    trainNoInfo = noInfoRate(train)
    holdNoInfo = noInfoRate(hold)

    # Test model on training set
    trainRes = testModel(model['forest'], train)
    trainAcc, trainUnsure = trainRes[0], trainRes[1]

    # Test model on holdout test set
    holdRes = testModel(model['forest'], hold)
    holdAcc, holdUnsure = holdRes[0], holdRes[1]

    # Print all info
    numAttrs, numTrees = model['numAttrs'], model['numTrees']
    print('')
    print('')
    print('///////////////////////////////////////////////////////////////////'\
        '/////////////')
    print('')
    print('                    Heart disease predictor: Random Forest')
    print('')
    print(f"      The training data contained {len(train)} rows. "\
        f"In the chosen model (evaluated") 
    print(f"      by accuracy on the validation set), {numAttrs} attributes "\
        "were randomly sampled")
    print(f"      at each decision point and {numTrees} trees were "\
        "built.")
    print('')
    print('      ** Training set predictions **')
    print(f"      No information rate: {trainNoInfo[0]}% of samples "\
        f"{trainNoInfo[1]}")
    print(f"      ACCURACY (on {len(train)} rows): {trainAcc}%")
    print(f"      INDECISIVE predictions: {trainUnsure}")
    print('')
    print('      ** Validation set predictions **')
    print(f"      No information rate: {valNoInfo[0]}% of samples "\
        f"{valNoInfo[1]}")
    print(f"      ACCURACY (on {len(val)} rows): {model['valAcc']}%")
    print(f"      INDECISIVE predictions: {model['valnUnsure']}")
    print('')
    print('      ** Holdout test set predictions **')
    print(f"      No information rate: {holdNoInfo[0]}% of samples "\
        f"{holdNoInfo[1]}")
    print(f"      ACCURACY (on {len(hold)} rows): {holdAcc}%")
    print(f"      INDECISIVE predictions: {holdUnsure}")
    print('')
    print('///////////////////////////////////////////////////////////////////'\
        '/////////////')
    print('')

################################################################################

# Calculates the no-information rate (prediction accuracy obtained by guessing
# the larger class every single time) given a dataframe.
def noInfoRate(df):
    hasDisease = len(df[df['HeartDisease'] == 1])
    noDisease = len(df[df['HeartDisease'] == 0])
    if hasDisease > noDisease:
        largerClass = hasDisease
        status = 'have heart disease.'
    else:
        largerClass = noDisease
        status = 'do not have heart disease.'
    
    rate = round(largerClass/len(df) * 100,2)
    return rate, status

################################################################################

# Runs a single row of data through the given decision tree and makes a 
# prediction for the HeartDisease attribute of that person.
def treePredict(df, node):
    # Base case - leaf node
    if node.left is None and node.right is None:
        return node.label
    
    # Recursive case
    if isinstance(node.splitPt, list) and node.splitAttr == 2:
        if df.loc['ChestPainType'].item() == node.splitPt[0] or \
            df.loc['ChestPainType'].item() == node.splitPt[1]:
            # ^^ need .item() to make it work because Python interprets
            # the truth value of the condition as a series, even though there
            # is only ever one value for df.loc['ChestPainType']
            return treePredict(df, node.left)
        else:
            return treePredict(df, node.right)
    else:
        if df.iloc[node.splitAttr].item() < node.splitPt:
            return treePredict(df, node.left)
        else:
            return treePredict(df, node.right)

# When going thru and predicting with tree, special case for ChestPainType and 
# node.splitPt being a list ---> this means we used a pairwise split, so rather 
# than splitting by that point we send the rows with CPType in the splitPt list
# to one side, and all other rows to the other side.

################################################################################

# Predicts whether the patient described by 'newData' will have heart disease
# using the given model. Returns the numbers of trees which voted yes and no
# respectively.
def predict(newData, model):
    yesCount, noCount = 0, 0
    for i in range(len(model['forest'])):
        if treePredict(newData, model['forest'][i].root) == 'Healthy':
            noCount += 1
        else:
            yesCount += 1

    return yesCount, noCount
