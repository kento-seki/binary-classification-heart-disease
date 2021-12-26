# A random forest algorithm to predict heart disease. 
# Based on the 'Heart Failure Prediction Dataset' taken from Kaggle 
# (https://www.kaggle.com/fedesoriano/heart-failure-prediction).

# Created by Kento Seki between 10th December and 22nd December 2021.

import pandas as pd
from randomForest import buildForest, catToNum, treePredict

while True:

    result = buildForest()
    forest, numTrees = result[0], result[1]

    while True:
        ans = input("Would you like to make a prediction on new data using this model? (y / n) ")
        if ans == 'n':
            break
        # Otherwise, answer was yes
        print('')
        print('Please provide the following data (see Dataset Definitions for attribute details).')
        age = int(input("      Age: "))
        sex = input("      Sex (M,F): ")
        chPain = input("      Chest pain type: (TA, ATA, NAP, ASY): ")
        restBP = int(input("      Resting blood pressure (integer) (mm Hg): "))
        chol = int(input("      Cholesterol (integer) (mm/dl): "))
        fastBS = int(input("      Fasting blood sugar (1 if more than 120 mg/dl, 0 otherwise): "))
        restECG = input("      Resting ECG results (Normal, ST, LVH): ")
        maxHR = int(input("      Maximum heart rate achieved (integer): "))
        exAng = input("      Exercise-induced angina (Y,N): ")
        oldpk = float(input("      ST segment depression (numeric): "))
        stSlope = input("      Slope of peak exercise ST segment (Up, Flat, Down): ")

        newData = {
            'Age': age, 'Sex': sex, 'ChestPainType': chPain, 'RestingBP': restBP,
            'Cholesterol': chol, 'FastingBS': fastBS, 'RestingECG': restECG,
            'MaxHR': maxHR, 'ExerciseAngina': exAng, 'Oldpeak': oldpk, 
            'ST_Slope': stSlope, 'HeartDisease': [-1]
        }

        newDf = pd.DataFrame(newData)
        newDf = catToNum(newDf)
        transDf = newDf.transpose()

        yesCount = 0
        noCount = 0
        for i in range(numTrees):
            if treePredict(transDf, forest[i].root) == 'Healthy':
                noCount += 1
            else:
                yesCount += 1

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
        print('////////////////////////////////////////////////////////////////////////////////')
        print('')
    
    print('')
    ans = input("Would you like to build a new model? (y / n) ")
    if ans == 'n':
        print('Ending program...')
        print('')
        exit()
    print('')
    # Otherwise, answer was yes