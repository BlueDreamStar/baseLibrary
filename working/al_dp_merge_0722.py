# input case
# case1:
# 3
# 2
# {{0,2,3},{2,0,1},{3,1,0}}

# case2:
# 4
# M(2,3,4,etc.)
# {{0,2,3,1},{2,0,2,3},{3,2,0,1},{1,3,1,0}}

import sys
import numpy as np

def solve(N, M, mapMatrix):
    if M == 1: return mapMatrix
    tempMatrix = np.zeros((N,N))
    idLeft = M // 2
    idRight = M - idLeft
    mapLeft = solve(N, idLeft, mapMatrix)
    mapRight = solve(N, idRight, mapMatrix)
    
    ### special: M <= 3
    if M <= 3: 
        for i in range(N):
            for j in range(N):
                tempIndex = np.intersect1d(np.nonzero(mapLeft[i,:]),np.nonzero(mapRight[:,j]))
                tempVector = mapLeft[i,tempIndex] + mapRight[tempIndex,j]
                tempMatrix[i,j] = np.min( tempVector)
        return tempMatrix
    
    for i in range(N):
        for j in range(N):
            tempVector = mapLeft[i,:] + mapRight[:,j]
            tempMatrix[i,j] = np.min( tempVector)
#             print(tempMatrix[i,j])
    return tempMatrix
    
# def main():
if __main__ == '__main__':
    N = int(input("input N:"))
    M = int(input("input M:"))
    mapStr = input("input map:")

    mapStrs = mapStr.replace(" ","").strip("{}").replace("},{",';')
    mapMatrix = np.array(np.matrix(mapStrs))
    
    if N <2 or N > 100: return
    if M < 2 or M > 1e6: return
    result = solve(N, M, mapMatrix)
    
    # stdout
    sys.stdout.write("result={\n")
    for i in range(N-1):
        sys.stdout.write("{")
        for j in range(N-1):
            sys.stdout.write("%d," %result[i,j])
        sys.stdout.write("%d},\n" %result[i,N-1])
    sys.stdout.write("{")
    for j in range(N-1):
        sys.stdout.write("%d," %result[N-1,j])
    sys.stdout.write("%d}\n}" %result[N-1,N-1])
