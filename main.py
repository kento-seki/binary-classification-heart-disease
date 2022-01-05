# A random forest algorithm to predict heart disease. 
# Based on the 'Heart Failure Prediction Dataset' taken from Kaggle 
# (https://www.kaggle.com/fedesoriano/heart-failure-prediction).

# Created by Kento Seki between 10th December and 22nd December 2021.
# Updated between 4th January 2022 and ________.

#  Definitions
#  * numAttrs - the number of attributes that will be randomly sampled and
#    considered as a splitting attribute at each node in each decision tree
#  * minNumAttrs/maxNumAttrs - the smallest and largest numAttrs values to
#    be tried when searching for the optimal model
#  * numTrees - the number of decision trees that comprise the random forest
#  * maxTrees - the largest number of trees per model to be tried when searching
#    for the optimal model

import pandas as pd
import random
from randomForest import buildForest, catToNum, modelInfo, treePredict

################################################################################
#                               MAIN FUNCTION

random.seed(98)

if __name__ == "__main__":

    # Set hyper-hyperparameters :)
    maxTrees = 10
    # ^ Note: maxTrees is not included (tries all forest sizes UP TO maxTrees)
    minNumAttrs, maxNumAttrs = 2, 6
    # ^ Note: minNumAttrs is included, while maxNumAttrs is not

    # Import data and map categorical variables to numbers
    # (ensure directory is /Workspace/HeartDisease)
    df = pd.read_csv('heart.csv')
    df = catToNum(df)

    # Create holdout test set and rename rows 0...len(hold)
    hold = df.iloc[ random.sample(range(len(df)), k = round(0.2 * len(df))) ]
    hold.set_axis(range(len(hold)), axis='index', inplace=True)
    leftover = pd.concat([df, hold]).drop_duplicates(keep = False)
    
    # Create validation set and rename rows 0...len(val)
    val = leftover.iloc[ random.sample(range(len(leftover)), \
        k = round(0.2 * len(df))) ]
    val.set_axis(range(len(val)), axis='index', inplace=True)

    # Make the remaining data the train set and rename rows 0...len(train)
    train = pd.concat([leftover, val]).drop_duplicates(keep = False)
    train.set_axis(range(len(train)), axis='index', inplace=True)

    # Try every combination of numAttr and numTrees to find the best model. 
    # Evaluate models by accuracy on validation set. Store the best so far 
    # in 'best'.
    best = {'forest': None, 'numTrees': None, 'numAttrs': None,
            'valAcc': 0, 'valnUnsure': None}

    # Loop over all possible numAttr values
    for numAttrs in range(minNumAttrs,maxNumAttrs):
        
        # Loop over reasonable numTrees values 
        for numTrees in range(1, maxTrees):
            model = buildForest(train, val, numTrees, numAttrs)
            # If the resulting model has a higher accuracy than the current 
            # best, save this model
            if model[1] > best['valAcc']:
                best['forest'] = model[0]
                best['numTrees'] = numTrees
                best['numAttrs'] = numAttrs
                best['valAcc'] = model[1]
                best['valnUnsure'] = model[2]
                print(f"Improved model found! Validation accuracy: "\
                    f"{model[1]}% and nUnsure: {model[2]}")
            # Update % progress
            done = (((numAttrs-minNumAttrs)*maxTrees) + numTrees)\
                    / ((maxNumAttrs - minNumAttrs) * maxTrees)
            perc = round(done * 100, 2)
            print(f'Progress: {perc}%', end="\r")
    
    # Now, 'best' contains the best model. Display model information
    modelInfo(train, val, hold, best)

    exit()

    while True:
        ans = input("""Would you like to make a prediction on new data 
        using this model? (y / n) """)

        if ans == 'n':
            break
        # If answer was yes, read in data and make prediction
        newData = readInput()
        yesCount, noCount = 0, 0
        for i in range(numTrees):
            if treePredict(newData, best['forest'][i].root) == 'Healthy':
                noCount += 1
            else:
                yesCount += 1

        printResult(yesCount, noCount)


################################################################################
#                               HELPER FUNCTIONS

# Prompts the user to input data for each of the required variables, reads the
# values and returns them as a dictionary.
def readInput():
    print('')
    print("""Please provide the following data (see Dataset Definitions 
    for attribute details).""")
    age = int(input("      Age: "))
    sex = input("      Sex (M,F): ")
    chPain = input("      Chest pain type: (TA, ATA, NAP, ASY): ")
    restBP = int(input("      Resting blood pressure (integer) (mm Hg): "))
    chol = int(input("      Cholesterol (integer) (mm/dl): "))
    fastBS = int(input("""      Fasting blood sugar (1 if more than 
    120 mg/dl, 0 otherwise): """))
    restECG = input("      Resting ECG results (Normal, ST, LVH): ")
    maxHR = int(input("      Maximum heart rate achieved (integer): "))
    exAng = input("      Exercise-induced angina (Y,N): ")
    oldpk = float(input("      ST segment depression (numeric): "))
    stSlope = input("""      Slope of peak exercise ST segment 
    (Up, Flat, Down): """)

    newData = {
        'Age': age, 'Sex': sex, 'ChestPainType': chPain, 'RestingBP': 
        restBP, 'Cholesterol': chol, 'FastingBS': fastBS, 'RestingECG': 
        restECG, 'MaxHR': maxHR, 'ExerciseAngina': exAng, 'Oldpeak': oldpk, 
        'ST_Slope': stSlope, 'HeartDisease': [-1]
    }

    newDf = pd.DataFrame(newData)
    newDf = catToNum(newDf)
    newDf = newDf.transpose()
    return newDf

# Prints the results of the predictions on the new data using the model that 
# was last built.
def printResult(yesCount, noCount):
    print('')
    print('                               *** RESULTS ***')
    print('')
    print(f"      Trees that voted NO heart disease: {noCount}")
    print(f"      Trees that voted YES heart disease: {yesCount}")
    print('')

    if noCount < yesCount:
        outcome = 'will'
    elif noCount > yesCount:
        outcome = 'will not'
    else:
        outcome = 'may or may not (inconclusive result)'

    print(f'      PREDICTION: the patient {outcome} suffer heart disease.')
    print('')
    print('///////////////////////////////////////////////////////////////'\
    '/////////////////')
    print('')
