# -*- coding: utf-8 -*-
# @Author: dreamBoy
# @Date:   2018-10-30 17:54:17
# @Email:  wpf2106@gmail.com
# @Desc:   Welcome to my world!
# @Motto:  Brave & Naive!
# @Last Modified by:   BlueDreamStar
# @Last Modified time: 2019-07-28 14:48:01
from __future__ import division 
import numpy as np
import sys
import random
import os
import datetime
from itertools import product
import math
import threading
from time import ctime,sleep
import scipy.stats as stats


##  append a str with timestamp into a given file
def log_str(log_file_name, string):
    output=str(datetime.datetime.now())+":"+str(string)
    open("%s" %log_file_name, "a").write(output+"\n")
    #print output

##  append a str into a giving file
def append_str(saveFile, string):
    open("%s" %saveFile, "a").write(string+"\n")
    

## create dir if not exists
def create_dir(target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        
## delete file if exists
def delete_file(target_file):
    if(os.path.exists(target_file)):
        os.remove(target_file)
        # print('File deleted successfully!')
    # else:
    #     print("File is not exists!")

## normlize one-dimensional array
def normalize_one_dimension(a):
    # print(a)
    # a = np.array(a)
    s = np.sum(a)
    if s != 0.0 and len(a) != 1:
        a = np.true_divide(a,s)
    return a

## normlize matrix
def normalize_matrix(matrix):
    tempmatrix = matrix.copy()
    tempsum = np.sum(tempmatrix, axis = 1)
    tempsum.shape = [len(tempsum), 1]
    return tempmatrix / tempsum

def acc_value(vector1, vector2):
    if len(vector1) != len(vector2):
        print("error")
        return
    tempvector = np.unique(vector1)
    templength = len( tempvector )
    tempvector = np.ones(templength)
    # tempvector = tempvector + np.bincount(vector2) + 1
    param = np.zeros(4)
    tempmatrix = np.ones( (templength, templength) )
    tempmatrix2 = np.ones( (templength, templength) )

    # print("tempvector")
    # print(tempvector)
    # print(tempmatrix.shape)
    # print()
    paramout = np.zeros(9)
    tempn = 0
    for i in range(len(vector1)):

        tempvector[vector2[i]] += 1   #### C
        # tempvector2[vector1[i]] += 1  #### P
        tempmatrix[ vector2[i], vector1[i]] += 1
        tempmatrix2[ vector2[i], vector1[i]] +=1
        for j in range( i + 1, len(vector1)):

            if vector1[i] == vector1[j]:
                if vector2[i] == vector2[j]:
                    param[0] += 1
                else:
                    param[2] += 1
            else:
                if vector2[i] == vector2[j]:
                    param[1] += 1
                else:
                    param[3] += 1
    paramout[0] = (param[0] + param[3]) * 1.0 / np.sum(param)  ## Rand Statistic
    paramout[1] = param[0] * 1.0 / (param[0] + param[1] + param[2]) ## Jaccard coefficient
    paramout[2] = np.sqrt( param[0] * param[0] * 1.0 / ( (param[0] + param[1]) * (param[0] + param[2]) ) ) ## Fowlkes and mallows index
    paramout[3] = param[0] * 1.0 / ( param[0] + param[2]) ## positive 
    paramout[4] = param[3] * 1.0 / ( param[1] + param[3]) ## negative
    paramout[5] = ( paramout[3] + paramout[4]) / 2  ## avg accuracy

    # print(tempmatrix)
    # print(tempvector)
    tempmatrix = ( tempmatrix.transpose() * 1.0 / tempvector ).transpose()
    tempmatrix = normalizedmatrix(tempmatrix)
    paramout[6] = (1.0 / np.log(len(tempvector)) ) * np.sum( tempmatrix * np.log( 1.0 / tempmatrix ) ) / len(tempvector)  ## avg entrop
    paramout[7] = np.sum(np.max(tempmatrix2, axis =1)) * 1.0 / len(vector1)

    tempPre = (tempmatrix2.transpose() * 1.0 / np.sum(tempmatrix2,axis =1) ).transpose()
    tempRec = tempmatrix2 * 1.0 / np.sum(tempmatrix2,axis=0) 
    tempF = 2 * (tempPre * tempRec) / ( tempPre + tempRec)
    tempFval = np.sum( 1.0 *np.max(tempF,axis=0) * np.sum( tempmatrix2, axis = 0) )/ np.sum(tempmatrix2)
    # tempEntroy = (1.0 / np.log(len(tempvector)) ) * ( np.sum( tempPre * np.log( 1 /tempPre) ) ).mean() /len(tempvector)
    paramout[8] = tempFval
    # paramout[9] = tempRec.mean()
    # print(tempPre)
    # print(tempRec)
    # print(paramout[8])
    # print(paramout[9])
    # print(tempF)
    # print( tempFval )  
    # print(tempEntroy)      
    return paramout
#### limit for exp
def limit_matrix( matrix , num=6):
    tempmat = matrix.copy()
    tempmax = np.max( tempmat, axis= 1)
    tempmax.shape = [ len(tempmax), 1]
    tempmat = tempmat - tempmax + num
    tempmask = tempmat <= 0
    tempmat[tempmask] = 0
    return tempmat 


#   compute the rmse value by two matrix
def rmse_value(matrixA, matrixB):

    differences = matrixA - matrixB                       #the DIFFERENCEs.
    differences_squared = differences ** 2                    #the SQUAREs of ^
    mean_of_differences_squared = differences_squared.mean()  #the MEAN of ^
    rmse_val = np.sqrt(mean_of_differences_squared)           #ROOT of ^

    return rmse_val

#   compute the mae value by two matrix
def mae_value(matrixA, matrixB):

	differences = matrixA - matrixB
	mae_val = np.abs(differences).mean()
	return mae_val

#   compute the exp value by dt (distance of two events) & W
def small_g(dt, W = 0.25):
	return W * np.exp(-W * dt)

#   compute the exp value by dt (distance of two events) & W
def big_g(dt, W = 0.25):
	return 1.0 - np.exp(-W * dt)

#   transfer the zeros to the minValue for a specific matrix
def zero_to_minvalue(matrix):
    mask = matrix <= 0
    tempmatrix = matrix[np.nonzero(np.abs(matrix))]
    if len(tempmatrix) == 0:
        tempvalue = 1E-100
    else:
        tempvalue = np.min(tempmatrix) / 1E-6
    matrix[mask] = tempvalue
    return matrix

#   transfer the zeros to ones for a specific matrix
def zero_to_one(matrix):
    tempmatrix = matrix.copy()
    mask = tempmatrix <= 0
    tempmatrix[mask] = 1
    return tempmatrix
