# A random forest algorithm to predict heart disease. 
# Based on the 'Heart Failure Prediction Dataset' taken from Kaggle 
# (https://www.kaggle.com/fedesoriano/heart-failure-prediction).

# Created by Kento Seki between 10th December and 22nd December 2021.

# THINGS TO THINK ABOUT: 
# - maximum depth as hyperparameter?


# Out-of-bag error - evaluation method:
# After each tree is created, take the rows of data that were NOT 
# used to build it and use it to make predictions of HeartDisease for those 
# rows. Record a tally of the number of TRUE and FALSE predictions for each
# row. Repeat this process for every tree.

from classes import Node, Tree
import pandas as pd
import random

def buildForest():
    # SET HYPERPARAMETERS (from input)
    hp = getHyperparams()
    numAttrs, numTrees = hp[0], hp[1]

    # BUILD MODEL
    df = pd.read_csv('heart.csv') # set directory to /Workspace/HeartDisease

    # Change categorical vars to numerical
    df = catToNum(df)

    # Randomise order of the dataset rows
    random.seed(97)
    rows = df.shape[0]
    shuff_rows = random.sample(range(rows),k = rows)
    shuff_df = df.iloc[shuff_rows] # need to use df.iloc[] to index rows

    # List to store trees (ith tree in forest[i])
    forest = []
    # Tally of out-of-bag predictions
    emptyTally = {'True': [0] * rows, 'False': [0] * rows, 'PredictedHD': '-'}
    tally = pd.DataFrame(emptyTally)

    ### OUTER FOR LOOP: FOR EACH TREE IN FOREST
    print('')
    print("I'm growing trees as fast as I can...")
    for i in range(numTrees):

        # Create bootstrapped dataframe
        btstrap_rows = random.choices(list(range(rows)), k = rows)
        btstrap_df = shuff_df.iloc[btstrap_rows]

        # Build decision tree
        # - Choose the best splitting attribute out of a random subset of size m
        #   (if no split provides improved impurity, don't split - this is a leaf)
        # - Add the split node to the tree
        newTree = Tree()
        newTree.root.data = btstrap_df
        newTree.root.unusedAttrs = list(range(11))
        forest.append(newTree)

        # Create the tree by recursively splitting nodes
        nodeSplit(newTree.root, numAttrs)

        # EVALUATION: make predictions for the out-of-bag data and add to tally
        outOfBag = pd.concat([df, btstrap_df, btstrap_df]).drop_duplicates(keep=False)
        for j in range(outOfBag.shape[0]):
            result = treePredict(outOfBag.iloc[j], forest[i].root)
            rowIndex = outOfBag.iloc[j].name
            if result == 'Healthy':
                tally.iloc[rowIndex, 1] = tally.iloc[rowIndex, 1] + 1
            elif result == 'Disease':
                tally.iloc[rowIndex, 0] = tally.iloc[rowIndex, 0] + 1
        
        # Update progress bar
        print('0|' + '=' * (i+1) + (numTrees - i - 1) * ' ' + f'|{numTrees}', end="\r")
    print('')

    ### Evaluate out-of-bag error from out-of-bag predictions tally
    nCorrect = 0
    nUnsure = 0
    for i in range(rows):
        if tally.iloc[i,0] > tally.iloc[i,1]:
            tally.iloc[i,2] = '1' # TRUE - has heart disease
        elif tally.iloc[i,0] < tally.iloc[i,1]:
            tally.iloc[i,2] = '0' # FALSE - does not have heart disease
        else:
            tally.iloc[i,2] = '-'
            nUnsure += 1

        if str(tally.iloc[i,2]) == str(df.iloc[i,11]):
            nCorrect += 1

    printModelInfo(df, rows, numAttrs, numTrees, nCorrect, nUnsure)
    return forest, numTrees

### KEEP GOING...
# + Evaluate performance with different hyperparameters using confusion matrix
# + Evaluate using ROC curve (and AUC?)
# See https://machinelearningmastery.com/classification-accuracy-is-not-enough-more-performance-measures-you-can-use/ 
# for more

################################################################################

# Get random forest hyperparameters from input.
def getHyperparams():
    print('')
    print("                         *** Hyperparameter Tuning ***")
    print('')
    numAttrs = int(input("Enter the number of attributes to sample: "))
    # ^How many attributes should be sampled as a candidate for each node split?
    # Less attributes -> trees more random ... more attributes -> less random
    # This is called 'mtry' in caret's ranger (random forest) method
    numTrees = int(input("Enter the number of trees: "))
    # ^How many trees should the forest consist of?
    return numAttrs, numTrees

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
                continue
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

# Recursively split the data at the given node (in preorder traversal order: 
# root->left->right)
def nodeSplit(node, numAttrs):
    #print(node.data)
    df = node.data
    # Base case for not splitting
    if len(df) <= 5 or len(node.unusedAttrs) == 0:
        node = assignLabel(node)
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

    # Make the best split found - send the data to left and right children
    if currImp <= minImp:
        node = assignLabel(node)
        return None
    elif df.columns[minAttr] == 'ChestPainType' and isinstance(minPt, list):
        # ...this means we did a pairwise split of ChestPainType
        # Store split info at this node
        node.splitAttr = minAttr
        node.splitPt = minPt
        newUnusedAttrs = node.unusedAttrs
        newUnusedAttrs.remove(minAttr)
        
        # Split data between two new children
        node.left = Node()
        node.left.data = df[(df['ChestPainType'] == minPt[0]) | (df['ChestPainType'] == minPt[1])]
        node.left.unusedAttrs = newUnusedAttrs
        node.right = Node()
        node.right.data = df[(df['ChestPainType'] != minPt[0]) & (df['ChestPainType'] != minPt[1])]
        node.right.unusedAttrs = newUnusedAttrs

    else:
        node.splitAttr = minAttr
        node.splitPt = minPt
        newUnusedAttrs = node.unusedAttrs
        newUnusedAttrs.remove(minAttr)
        attrName = df.columns[minAttr]
        
        # Split data between two new children
        node.left = Node()
        node.left.data = df[df[attrName] < minPt]
        node.left.unusedAttrs = newUnusedAttrs
        node.right = Node()
        node.right.data = df[df[attrName] > minPt]
        node.right.unusedAttrs = newUnusedAttrs

    nodeSplit(node.left, numAttrs)
    nodeSplit(node.right, numAttrs)

################################################################################

# Runs a single row of data through the given decision tree and makes a 
# prediction for the HeartDisease attribute of that person.
def treePredict(df, node):
    # Base case - leaf node
    if node.left is None and node.right is None:
        return node.label
    
    # Recursive case
    if isinstance(node.splitPt, list) and node.splitAttr == 2:
        if df.loc['ChestPainType'].item() == node.splitPt[0] or df.loc['ChestPainType'].item() == node.splitPt[1]:
            # ^^ need .item() to make it work because Python interprets
            # the truth value of the condition as a series, even though there
            # is only ever one value for df.loc['ChestPainType']
            return treePredict(df, node.left)
        else:
            return treePredict(df, node.right)
    else:
        #print(df.iloc[node.splitAttr])
        #print('-')
        #return##
        if df.iloc[node.splitAttr].item() < node.splitPt:
            return treePredict(df, node.left)
        else:
            return treePredict(df, node.right)
        

# When going thru and predicting with tree, special case for ChestPainType and 
# node.splitPt being a list ---> this means we used pairwise split, so rather 
# than splitting by that point we send the rows with CPType in the splitPt list
# to one side, and all other rows to the other side.

################################################################################

def printModelInfo(df, rows, numAttrs, numTrees, nCorrect, nUnsure):
    # Calculate no-info rate for comparison
    hasDisease = len(df[df['HeartDisease'] == 1])
    noDisease = len(df[df['HeartDisease'] == 0])
    if hasDisease > noDisease:
        largerClass = hasDisease
        status = 'have heart disease.'
    else:
        largerClass = noDisease
        status = 'do not have heart disease.'

    print('')
    print('')
    print('////////////////////////////////////////////////////////////////////////////////')
    print('')
    print('                    Heart disease predictor: Random Forest')
    print('')
    print(f"      The given dataset contained {rows} rows. {numAttrs} attributes were randomly")
    print(f"      sampled at each decision point and {numTrees} trees were built.")
    print('')
    print(f'      No information rate: {round(largerClass/rows * 100,2)}% of samples {status}')
    print('')
    print(f"      ACCURACY of predictions on out-of-bag samples: {round(nCorrect/rows * 100,2)}%")
    print(f"      INDECISIVE predictions: {nUnsure}")
    print('')
    print('////////////////////////////////////////////////////////////////////////////////')
    print('')