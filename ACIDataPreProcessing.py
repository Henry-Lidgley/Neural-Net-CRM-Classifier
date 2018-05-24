#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @author: henrylidgley 2018

# ACI Project Data Preprocessing - File 1 of 2

# Import APIs
import numpy as np
import pandas as pd

# Small dataset and target labels can be found at http://www.kdd.org/kdd-cup/view/kdd-cup-2009/Data

# Importing the dataset
orangeDataset = pd.read_csv("orange_small_train.data", sep='\t')

# =============================================================================
# Encoding numerical variables
# =============================================================================

# separate numerical features
numFeaturesArray = orangeDataset.iloc[:, 0:190].values
# Convert array object to dataframe
numFeaturesDF = pd.DataFrame(numFeaturesArray) 

# Name numerical dataframe columns
i = 1
while (i < 191):
    oldColumnName = i - 1
    newColumnName = 'Var' + str(i)
    numFeaturesDF.rename(columns = {oldColumnName:newColumnName}, inplace=True)
    i = i + 1

# The below commented section is only required once to ID which features to drop
"""# Obtain numerical variable value frequencies
i = 1

numericalVariables = {}

while (i < 191):
    numericalVariables[i] = pd.DataFrame()
    i = i + 1
    
i = 1

while (i < 191):

    numericalVariables[i] = numFeaturesDF.iloc[:, (i - 1)].values
    numericalVariables[i]  = pd.DataFrame(numericalVariables[i]) 
    numericalVariables[i].columns = ['Var' + str(i)]
    numericalVariables[i]  = numericalVariables[i].fillna('Empty').groupby('Var' + str(i)).size()
    numericalVariables[i]  = pd.DataFrame(numericalVariables[i] ) 
    numericalVariables[i]  = numericalVariables[i].sort_values(ascending=False, by=[0])

    i = i + 1"""
    
# Delete empty numerical variables and those with fewer than 8 values due to lack of variance
numFeaturesDF = numFeaturesDF.drop(numFeaturesDF.columns[[1, 3, 7, 10, 14, 18, 19, 25, 26, 28, 30, 31, 33, 38, 41, 47, 48, 51, 53, 54, 66, 78, 81, 86, 89, 92, 99, 109, 115, 117, 121, 129, 137, 140, 141, 142, 146, 166, 168, 172, 174, 184]], axis=1)

# Replace NaN values with 0
numFeaturesDF = numFeaturesDF.fillna(0)

# =============================================================================
# Encoding categorical features
# =============================================================================

# Separate category features
catFeaturesArray = orangeDataset.iloc[:, 190:230].values

# Convert array object to dataframe
catFeaturesDF = pd.DataFrame(catFeaturesArray) 

# Name dataframe columns
catFeaturesDF.columns = ['Var191', 'Var192', 'Var193', 'Var194', 'Var195', 'Var196', 'Var197', 'Var198', 'Var199', 'Var200', 'Var201', 'Var202', 'Var203', 'Var204', 'Var205', 'Var206', 'Var207', 'Var208', 'Var209', 'Var210', 'Var211', 'Var212', 'Var213', 'Var214', 'Var215', 'Var216', 'Var217', 'Var218', 'Var219', 'Var220', 'Var221', 'Var222', 'Var223', 'Var224', 'Var225', 'Var226', 'Var227', 'Var228', 'Var229', 'Var230']

# Create dictionary to hold categorical variable frequencies
i = 191
categoryVariables = {}
while (i < 231):
    categoryVariables[i] = pd.DataFrame()
    i = i + 1
    
# Include NaN values in groupby 
np.set_printoptions(threshold = np.nan)   
    
# Sort values in each variable by frequency      
i = 191
while (i < 231):

    categoryVariables[i] = catFeaturesDF.iloc[:, (i - 191)].values
    categoryVariables[i]  = pd.DataFrame(categoryVariables[i]) 
    categoryVariables[i].columns = ['Var' + str(i)]
    categoryVariables[i]  = categoryVariables[i].fillna('Empty').groupby('Var' + str(i)).size()
    categoryVariables[i]  = pd.DataFrame(categoryVariables[i] ) 
    categoryVariables[i]  = categoryVariables[i].sort_values(ascending=False, by=[0])

    i = i + 1

# Set category variable frequency thresholds
catVarFreqThresholds = {}

i = 0
while (i < 40):
    catVarFreqThresholds[i] = pd.DataFrame()
    i = i + 1

# For 8 values, starting at variable 191
catVarFreqThresholds[0] = 0
catVarFreqThresholds[1] = 370
catVarFreqThresholds[2] = 400
catVarFreqThresholds[3] = 0
catVarFreqThresholds[4] = 23
catVarFreqThresholds[5] = 0
catVarFreqThresholds[6] = 622
catVarFreqThresholds[7] = 229
catVarFreqThresholds[8] = 550
catVarFreqThresholds[9] = 20
# Variable 201
catVarFreqThresholds[10] = 0
catVarFreqThresholds[11] = 105
catVarFreqThresholds[12] = 0
catVarFreqThresholds[13] = 1033
catVarFreqThresholds[14] = 0
catVarFreqThresholds[15] = 1510
catVarFreqThresholds[16] = 70
catVarFreqThresholds[17] = 0
catVarFreqThresholds[18] = 0
catVarFreqThresholds[19] = 0
# Variable 211
catVarFreqThresholds[20] = 0
catVarFreqThresholds[21] = 690
catVarFreqThresholds[22] = 0
catVarFreqThresholds[23] = 23
catVarFreqThresholds[24] = 0
catVarFreqThresholds[25] = 1000
catVarFreqThresholds[26] = 152
catVarFreqThresholds[27] = 0
catVarFreqThresholds[28] = 80
catVarFreqThresholds[29] = 229
# Variable 221
catVarFreqThresholds[30] = 0
catVarFreqThresholds[31] = 229
catVarFreqThresholds[32] = 0
catVarFreqThresholds[33] = 0
catVarFreqThresholds[34] = 0
catVarFreqThresholds[35] = 2190
catVarFreqThresholds[36] = 0
catVarFreqThresholds[37] = 900
catVarFreqThresholds[38] = 0
catVarFreqThresholds[39] = 0

# Drop variables containing little variance using frequency thresholds & create dummy variables
def groupCatFeatures(df, threshold, column, prefix, normalize=False):
    freqencies = df[column].value_counts( sort=False, normalize=normalize)
    idx = freqencies[freqencies < threshold].index
    tmp = df
    if idx.shape[0] > 0:
        tmp[column] = df[column].replace(idx, 'Uncommon')
    else:
        tmp = df
    d = pd.get_dummies(tmp, columns=[column], prefix=prefix, dummy_na=True)
        
    return d

# Call groupCatFeatures method for all cat variables and name columns
i = 191
while (i < 231):
    catFeaturesDF = groupCatFeatures(df = catFeaturesDF, threshold = catVarFreqThresholds[i - 191], column = 'Var' + str(i), prefix = 'Var' + str(i)) 
    i = i + 1

# Delete NaN column to avoid dummy variable trap
i = 191
while (i < 231):
    del catFeaturesDF['Var' + str(i) + '_nan']
    i = i + 1

# =============================================================================
# Concatinating the categorical and numerical features with target labels
# =============================================================================

# Concatonating numerical and categorical features
X = pd.concat([numFeaturesDF, catFeaturesDF], axis=1)

# Exporting dataset to csv
X.to_csv('orangeXdata.csv', sep='\t')