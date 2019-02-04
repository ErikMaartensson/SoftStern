from __future__ import division
from sage.all import *

import numpy as np
import heapq as hq
from timeit import default_timer as timer

# GENERATOR AND PARITY-CHECK MATRICES BEGIN
def getExtendedGolayGeneratorMatrix():
    
    G = np.matrix(
       [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1]], dtype = 'int')
    
    return G

# Returns a random generator matrix of size K*N in systematic form
def getRandomGeneratorMatrix(K, N):
    
    I = np.eye(K, dtype = 'int')
    P = np.matrix(np.random.randint(2, size = (K, N - K)), dtype = 'int')
    G = np.bmat([I, P])
    return G

    # return np.matrix(np.random.randint(2, size = (K, N)))


# HYBRID LDPC PARITY-CHECK MATRICES BEGIN
def getLDPC64ParityCheckMatrix():
    M = 16
    I = np.identity(M, dtype = 'int')
    Z = np.zeros((M, M), dtype = 'int')
    
    # Row 1
    H11 = I + np.roll(I, 7, axis = 1)
    H12 = np.roll(I, 2, axis = 1)
    H13 = np.roll(I, 14, axis = 1)
    H14 = np.roll(I, 6, axis = 1)
    H15 = Z.copy()
    H16 = I.copy()
    H17 = np.roll(I, 13, axis = 1)
    H18 = I.copy()
    
    # Row 2
    H21 = np.roll(I, 6, axis = 1)
    H22 = I + np.roll(I, 15, axis = 1)
    H23 = I.copy()
    H24 = np.roll(I, 1, axis = 1)
    H25 = I.copy()
    H26 = Z.copy()
    H27 = I.copy()
    H28 = np.roll(I, 7, axis = 1)
    
    # Row 3
    H31 = np.roll(I, 4, axis = 1)
    H32 = np.roll(I, 1, axis = 1)
    H33 = I + np.roll(I, 15, axis = 1)
    H34 = np.roll(I, 14, axis = 1)
    H35 = np.roll(I, 11, axis = 1)
    H36 = I.copy()
    H37 = Z.copy()
    H38 = np.roll(I, 3, axis = 1)
    
    # Row 4
    H41 = I.copy()
    H42 = np.roll(I, 1, axis = 1)
    H43 = np.roll(I, 9, axis = 1)
    H44 = I + np.roll(I, 13, axis = 1)
    H45 = np.roll(I, 14, axis = 1)
    H46 = np.roll(I, 1, axis = 1)
    H47 = I.copy()
    H48 = Z.copy()
    
    H = np.bmat([[H11, H12, H13, H14, H15, H16, H17, H18],
                 [H21, H22, H23, H24, H25, H26, H27, H28],
                 [H31, H32, H33, H34, H35, H36, H37, H38],
                 [H41, H42, H43, H44, H45, H46, H47, H48]])
       
    return H

def getLDPC128ParityCheckMatrix():
    M = 32
    I = np.identity(M, dtype = 'int')
    Z = np.zeros((M, M), dtype = 'int')
    
    # Row 1
    H11 = I + np.roll(I, 31, axis = 1)
    H12 = np.roll(I, 15, axis = 1)
    H13 = np.roll(I, 25, axis = 1)
    H14 = I.copy()
    H15 = Z.copy()
    H16 = np.roll(I, 20, axis = 1)
    H17 = np.roll(I, 12, axis = 1)
    H18 = I.copy()
    
    # Row 2
    H21 = np.roll(I, 28, axis = 1)
    H22 = I + np.roll(I, 30, axis = 1)
    H23 = np.roll(I, 29, axis = 1)
    H24 = np.roll(I, 24, axis = 1)
    H25 = I.copy()
    H26 = Z.copy()
    H27 = np.roll(I, 1, axis = 1)
    H28 = np.roll(I, 20, axis = 1)
    
    # Row 3
    H31 = np.roll(I, 8, axis = 1)
    H32 = I.copy()
    H33 = I + np.roll(I, 28, axis = 1)
    H34 = np.roll(I, 1, axis = 1)
    H35 = np.roll(I, 29, axis = 1)
    H36 = I.copy()
    H37 = Z.copy()
    H38 = np.roll(I, 21, axis = 1)
    
    # Row 4
    H41 = np.roll(I, 18, axis = 1)
    H42 = np.roll(I, 30, axis = 1)
    H43 = I.copy()
    H44 = I + np.roll(I, 30, axis = 1)
    H45 = np.roll(I, 25, axis = 1)
    H46 = np.roll(I, 26, axis = 1)
    H47 = I.copy()
    H48 = Z.copy()
    
    H = np.bmat([[H11, H12, H13, H14, H15, H16, H17, H18],
                 [H21, H22, H23, H24, H25, H26, H27, H28],
                 [H31, H32, H33, H34, H35, H36, H37, H38],
                 [H41, H42, H43, H44, H45, H46, H47, H48]])  
    
    return H

def getLDPC256ParityCheckMatrix():
    M = 64
    I = np.identity(M, dtype = 'int')
    Z = np.zeros((M, M), dtype = 'int')
    
    # Row 1
    H11 = I + np.roll(I, 63, axis = 1)
    H12 = np.roll(I, 30, axis = 1)
    H13 = np.roll(I, 50, axis = 1)
    H14 = np.roll(I, 25, axis = 1)
    H15 = Z.copy()
    H16 = np.roll(I, 43, axis = 1)
    H17 = np.roll(I, 62, axis = 1)
    H18 = I.copy()
    
    # Row 2
    H21 = np.roll(I, 56, axis = 1)
    H22 = I + np.roll(I, 61, axis = 1)
    H23 = np.roll(I, 50, axis = 1)
    H24 = np.roll(I, 23, axis = 1)
    H25 = I.copy()
    H26 = Z.copy()
    H27 = np.roll(I, 37, axis = 1)
    H28 = np.roll(I, 26, axis = 1)
    
    # Row 3
    H31 = np.roll(I, 16, axis = 1)
    H32 = I.copy()
    H33 = I + np.roll(I, 55, axis = 1)
    H34 = np.roll(I, 27, axis = 1)
    H35 = np.roll(I, 56, axis = 1)
    H36 = I.copy()
    H37 = Z.copy()
    H38 = np.roll(I, 43, axis = 1)
    
    # Row 4
    H41 = np.roll(I, 35, axis = 1)
    H42 = np.roll(I, 56, axis = 1)
    H43 = np.roll(I, 62, axis = 1)
    H44 = I + np.roll(I, 11, axis = 1)
    H45 = np.roll(I, 58, axis = 1)
    H46 = np.roll(I, 3, axis = 1)
    H47 = I.copy()
    H48 = Z.copy()
    
    H = np.bmat([[H11, H12, H13, H14, H15, H16, H17, H18],
                 [H21, H22, H23, H24, H25, H26, H27, H28],
                 [H31, H32, H33, H34, H35, H36, H37, H38],
                 [H41, H42, H43, H44, H45, H46, H47, H48]])  
    
    return H
# HYBRID LDPC PARITY-CHECK MATRICES END


# HYBRID LDPC GENERATOR MATRICES BEGIN
def getLDPC64GeneratorMatrix():
    H = getLDPC64ParityCheckMatrix()
    
    I = np.identity(64, dtype = 'int')
    
    Q = matrix(GF(2), H[:, 0:64])
    P = matrix(GF(2), H[:, 64:128])
    
    W = np.array(transpose(P.inverse()*Q))
    
    G = np.bmat([[I, W]])
    
    return G

def getLDPC128GeneratorMatrix():
    H = getLDPC128ParityCheckMatrix()
    
    I = np.identity(128, dtype = 'int')
    
    Q = matrix(GF(2), H[:, 0:128])
    P = matrix(GF(2), H[:, 128:256])
    
    W = np.array(transpose(P.inverse()*Q))
    
    G = np.bmat([[I, W]])
    
    return G

def getLDPC256GeneratorMatrix():
    H = getLDPC256ParityCheckMatrix()
    
    I = np.identity(256, dtype = 'int')
    
    Q = matrix(GF(2), H[:, 0:256])
    P = matrix(GF(2), H[:, 256:512])
    
    W = np.array(transpose(P.inverse()*Q))
    
    G = np.bmat([[I, W]])
    
    return G
# HYBRID LDPC GENERATOR MATRICES END



# GENERATOR AND PARITY-CHECK MATRICES END

def encrypt(u, G, N, sigma):
    # Encodes the message
    c = np.array(np.mod(u*G, 2)).squeeze()
    v = 1 - 2*c.astype('float') # Transforms the encoding
    # Adds random noice to the vector v
    gaussian = RealDistribution('gaussian', sigma)
    # gaussian.set_seed(0.0) # # To make results reproducible, might want to change this
    r = v + [gaussian.get_random_element() for i in xrange(N)]
    
    # Hard-codes in a specific noice vector
    # r = np.array([0.843729958623899, -0.805925056568809, 1.57999998093339, -0.149148717337394, 0.574142767658874, 0.692213657241573, 1.07337878801990, -1.93055392814279, -1.56778457457770, 1.14053569724447,    0.899776799933957,  2.64396123919530,   1.23896886133823, -0.684705425440007, 0.0724143541578204, -0.231475557145293, 0.180045642651507, -0.803167846346245, -1.35328940557060, -2.82475788374105, 0.131862782719346, 0.0887631338282713, -1.23010963207342, -0.346875583617594])
    return r
    
def invOrder(v, I, shift = 0):
    v_copy = v.copy()
    for i in xrange(len(I)):
        v[I[i]] = v_copy[i + shift]
        
def getL1(K, l, T, llr, HT):
    # Gets the first candidate vectors for S1, called L1
    
    # A matrix with the T most probable bit vectors and the corresponding syndromes 
    L1 = np.zeros ((T, floor((K + l)/2) + l), dtype = 'int')
    
    # A matrix with the candidates for the next most probable vector, the bit to be changed for the next vector
    C1 = np.zeros ((T, floor((K + l)/2) + 1), dtype = 'int')
    
    # The llr sum for the next vector
    C1_llrsum = np.zeros((T, 1))
    C1_llrsum [0] = llr[0] # The corresponding llr sum is llr[0]
    
    for m in xrange(1, T):
        
        index = C1[0, -1] # The index for the bit position that should be 1 in the next candidate vector
#        vector = C1[0, 0:(K + l)/2].copy() # The new candidate vector
        vector = C1[0, 0:floor((K + l)/2)].copy() # The new candidate vector
        vector[index] = 1        
        L1[m, 0:floor((K + l)/2)] = vector # Add the vector to list of candidates
        L1[m, floor((K + l)/2):floor((K + l)/2) + l] = vector*HT[0:floor((K + l)/2), :] # Add the corresponding syndrome
        # Modify the parent node and add the child node in C1
        
        if index < floor((K + l)/2) - 1: # Add the parent and child node to list of candidates
            # Child node
            C1[m, 0:floor((K + l)/2)] = vector # Adds the new vector to the set of candidates
            C1[m, -1] = index + 1 # Sets its next index
            C1_llrsum[m] = C1_llrsum[0] + llr[index + 1] # Sets its llr value

            # Parent node
            C1[0, -1] = index + 1 # Updates next index for parent node
            C1_llrsum[0] = C1_llrsum[0] + llr[index + 1] - llr[index] # Updates llr sum for parent node
        else:
            C1_llrsum[0] = sys.maxint # Makes it impossible to choose a vector with an index outside of the vector size
            C1_llrsum[m] = sys.maxint # Also make it impossible to choose the child node!
        
        # Sort the list of candidates according to the llr values
        Illr1 = np.argsort(C1_llrsum[0:m].squeeze())
        C1_llrsum[0:m] = C1_llrsum[Illr1]
        C1[0:m, :] = C1[Illr1, :]
    return L1

def getL2(K, l, T, llr, HT, syndrome):
    # Gets the first candidate vectors for S2, called L2
    
    # A matrix with the T most probable bit vectors and the corresponding syndromes 
    L2 = np.zeros ((T, ceil((K + l)/2) + l), dtype = 'int')
    L2 [0, ceil((K + l)/2):ceil((K + l)/2) + l] = syndrome.copy()
    
    # A matrix with the candidates for the next most probable vector, the bit to be changed for the next vector
    C2 = np.zeros ((T, ceil((K + l)/2) + 1), dtype = 'int')
    
    # The llr sum for the next vector
    C2_llrsum = np.zeros((T, 1))
    C2_llrsum [0] = llr[0] # The corresponding llr sum is llr[0]
    
    for m in xrange(1, T):
        
        index = C2[0, -1] # The index for the bit position that should be 1 in the next candidate vector
        vector = C2[0, 0:ceil((K + l)/2)].copy() # The new candidate vector
        vector[index] = 1        
        L2[m, 0:ceil((K + l)/2)] = vector # Add the vector to list of candidates
        L2[m, ceil((K + l)/2):ceil((K + l)/2) + l] = vector*HT[floor((K + l)/2):K + l, :]  + syndrome
        # Add the corresponding syndrome
        # Modify the parent node and add the child node in C2
        
        if index < ceil((K + l)/2) - 1:
            # Child node
            C2[m, 0:ceil((K + l)/2)] = vector # Adds the new vector to the set of candidates
            C2[m, -1] = index + 1 # Sets its next index
            C2_llrsum[m] = C2_llrsum[0] + llr[index + 1] # Sets its llr value

            # Parent node
            C2[0, -1] = index + 1 # Updates next index for parent node
            C2_llrsum[0] = C2_llrsum[0] + llr[index + 1] - llr[index] # Updates llr sum for parent node
        else:
            C2_llrsum[0] = sys.maxint # Makes it impossible to choose a vector with an index outside of the vector size
            C2_llrsum[m] = sys.maxint # Also make it impossible to choose the child node!
        
        # Sort the list of candidates according to the llr values
        Illr1 = np.argsort(C2_llrsum[0:m].squeeze())
        C2_llrsum[0:m] = C2_llrsum[Illr1]
        C2[0:m, :] = C2[Illr1, :]
    return L2

def decrypt(r, G, N, K, l, T, sigma, time = False):
    # STARTS TRANSFORMATION OF THE PROBLEM

    if time:
        start = timer()
    
    I = np.argsort([-abs(r)]) # Not optimal to negate the vector
    r_prime = r[I].squeeze()
    G_prime = G[:, I].squeeze()

    # Start attempts to find an inverse, quite ugly solution right now
    attempts = 0
    while True:
        I2 = np.random.permutation(range(K + l))
        G_temp = matrix(GF(2), G_prime[:, I2[0:K]])
        if G_temp.is_invertible():
            G_inv = G_temp.inverse()
            break
        else:
            attempts = attempts + 1
            # print "Number of attempts at finding an inverse: " + str(attempts)
            
            # Give up finding an inverse after 200. Return a random message
            if attempts > 200:
                print ("Fails to find an invertible submatrix among k + l most reliable positions!")
                u_hat = np.random.randint(2, size = K)
                return u_hat
    # End attempts to find an inverse

    r_prime[0:K + l] = r_prime[I2].squeeze()
    G_prime[:, 0:K + l] = G_prime[:, I2].squeeze()

    G_star = G_inv*G_prime

    # Modifies the r values such that the first K are positive
    u_prime = r_prime[0:K] < 0
    c_prime = np.array(u_prime*G_star).squeeze()
    r_star = np.array(1 - 2*c_prime.astype('float'))*r_prime

    # Generates the final r values and calculates the syndrome for the l middle values
    r_final = r_star[0:K+l].squeeze().copy()
    syndrome = r_final[K:K+l] < 0
    r_final[K:K+l] = abs(r_final[K:K+l])

    # Generates the transpose of the parity-check matrix
    G_final = G_star[:, 0:K + l]
    b1 = G_final[:, K:K + l]
    b2 = np.identity(l, dtype = 'int')
    HT = np.bmat('b1; b2')
    
    # Sorts the first floor((K + l)/2) positions S1 and the last ceil((K + l)/2) positions S2 according to their r values
    IS1 = np.argsort(r_final[0:(K + l)//2])
    IS2 = np.argsort(r_final[(K + l)//2:K + l])
    IS2 = IS2 + (K + l)//2

    # Modifies the corresponding rows in HT and positions in r, syndrome and calculates the llr values
    HT[0:(K + l)//2, :] = HT[IS1, :]
    HT[(K + l)//2:K + l, :] = HT[IS2, :]
    
    # print HT

    r_final[0:(K + l)//2] = r_final[IS1]
    r_final[(K + l)//2:K + l] = r_final[IS2]

    llr_final = 2*r_final/sigma^2
    
    if time:
        print "Time until transformation done: " + str(timer() - start) + " s"

    # ENDS TRANSFORMATION OF THE PROBLEM


    # STARTS SOLVING THE MAIN PROBLEM

    # Creates the lists of the most probable bit patterns and the corresponding syndromes
    L1 = getL1(K, l, T, llr_final[0:(K + l)//2], HT)
    L2 = getL2(K, l, T, llr_final[(K + l)//2:K + l], HT, syndrome)
    
    if time:
        print "Time until lists found: " + str(timer() - start) + " s"
    
    # Variables for most probable vector yet and its distance
    r_min = 1e10
    c_min = np.ones((1, N), dtype = 'int')

    # Collides the lists and evaluates the collision's distance to r
    
    # Puts list 1 in a dictionary
    cand = {}
    for i in xrange(np.shape(L1)[0]):
        key = tuple(L1[i, (K + l)//2:(K + l)//2 + l]) # questionable - should this be changed?
        if not key in cand:
            cand[key] = [L1[i, 0:(K + l)//2]]
        else:
            cand[key].append(L1[i, 0:(K + l)//2])

    # print "Number of unique syndromes in list 1: " + str(len(cand))
    
    # Puts list 2 in a dictionary too - for debugging purposes only...
    # cand2 = {}
    # for i in xrange(np.shape(L2)[0]):
    #     key = tuple(L2[i, ceil((K + l)/2):ceil((K + l)/2) + l]) # questionable - should this be changed?
    #     if not key in cand2:
    #         cand2[key] = [L2[i, 0:ceil((K + l)/2) + 1]]
    #     else:
    #         cand2[key].append(L2[i, 0:ceil((K + l)/2) + 1])

    # print "Number of unique syndromes in list 2: " + str(len(cand2))
    
    # numberOfUniqueCollisions = 0
    # totalNumberOfCollisions = 0
    
    # Loops trough list 2 to find collisions
    for i in xrange(L2.shape[0]):
        # print i
        key = tuple(L2[i, ceil((K + l)/2):ceil((K + l)/2) + l]) # questionable - should this be changed?
        if key in cand:            
            # numberOfUniqueCollisions += 1
            # totalNumberOfCollisions += len(cand[key])
            # print cand[key]
            # print len(cand[key])
            for vec1 in cand[key]:
                u0 = np.append(vec1, L2[i, 0:ceil((K + l)/2) + 1])

                invOrder(u0, IS1) # Ugly solution
                invOrder(u0, IS2, (K + l)//2) # Ugly solution
                # u0[IS1] = u0[0:(K + l)/2]    
                # u0[IS2] = u0[(K + l)/2:K + l]
                u0 = u0[0:K]
                c0 = u0 * G_star
                v0 = 1 - 2*c0.astype('float')

                r_temp = np.linalg.norm(r_star - v0)
                if r_temp < r_min:
                    r_min = r_temp
                    c_min = c0
    
    # print "Number of unique collisions: " + str(numberOfUniqueCollisions)    
    # print "Total number of collisions: " + str(totalNumberOfCollisions)
    
    if time:
        print "Time until lists collided: " + str(timer() - start) + " s"
    # ENDS SOLVING THE MAIN PROBLEM



    # STARTS INVERSE TRANSFORMATION OF THE PROBLEM
    c_min = c_min.A1.astype('bool')
    c_hat = np.logical_xor(r_star != r_prime, c_min)

    # Inverts the permutation I2
    # c_hat[I2] = c_hat[0:K + l] # Want to write it like this...
    invOrder(c_hat, I2) # Ugly solution...

    # Inverts the permutation I
    # c_hat[I] = c_hat # Want to write it like this...
    invOrder(c_hat, I.squeeze()) # Ugly solution...

    # Finds the corresponding original message
    u_hat = c_hat[0:K].astype('int')

    if time:
        print "Total time: " + str(timer() - start) + " s"
        
    # ENDS INVERSE TRANSFORMATION OF THE PROBLEM
    return u_hat

def correctGolayDecryption(sigma, l, T):
    G = getExtendedGolayGeneratorMatrix()
    K, N = np.shape(G)
    
    for i in xrange(2^K - 1):
        u = np.array([int(i) for i in bin(6)[2:].zfill(K)])
        r = encrypt(u, G, N, sigma) # Encrypts the message randomly
        u_hat = decrypt(r, G, N, K, l, T, sigma)

        if sum(u_hat - u) != 0:
            return False
    return True

def testLDPC64Decryption(sigma, l, T, sim):
    G = getLDPC64GeneratorMatrix()
    K, N = np.shape(G)
    correct = 0
    
    for i in xrange(sim):
        # print i
        u = np.random.randint(2, size = K)
        r = encrypt(u, G, N, sigma) # Encrypts the message randomly
        u_hat = decrypt(r, G, N, K, l, T, sigma, time = 1)

        if sum(u_hat - u) == 0:
            correct = correct + 1
    return correct/float(sim)

##### Functions for simulating the P(A) values begins here #####

# Creates the list of the most probable bit patterns
def bitPatternList(llr, T, n):
    # Gives the T most probable bit patterns for the n llr values
    
    # A matrix with the T most probable bit vectors
    L = np.zeros ((T, n), dtype = 'int')
    
    # A matrix with the candidates for the next most probable vector, the bit to be changed for the next vector
    C = np.zeros ((T, n + 1), dtype = 'int')
    
    # The llr sum for the next vector
    C_llrsum = np.zeros((T, 1))
    C_llrsum [0] = llr[0] # The corresponding llr sum is llr[0]
    
    for m in xrange(1, T):
        
        index = C[0, -1] # The index for the bit position that should be 1 in the next candidate vector
        vector = C[0, 0:n].copy() # The new candidate vector
        vector[index] = 1        
        L[m, 0:n] = vector # Add the vector to list of candidates
        # Modify the parent node and add the child node in C
        
        if index < n - 1: # Add the parent and child node to list of candidates
            # Child node
            C[m, 0:n] = vector # Adds the new vector to the set of candidates
            C[m, -1] = index + 1 # Sets its next index
            C_llrsum[m] = C_llrsum[0] + llr[index + 1] # Sets its llr value

            # Parent node
            C[0, -1] = index + 1 # Updates next index for parent node
            C_llrsum[0] = C_llrsum[0] + llr[index + 1] - llr[index] # Updates llr sum for parent node
        else:
            C_llrsum[0] = sys.maxint # Makes it impossible to choose a vector with an index outside of the vector size
            C_llrsum[m] = sys.maxint # Also make it impossible to pick the child node!
        
        # Sort the list of candidates according to the llr values
        Illr = np.argsort(C_llrsum[0:m].squeeze())
        C_llrsum[0:m] = C_llrsum[Illr]
        C[0:m, :] = C[Illr, :]
    return L
    
def bitPatternListSum(llr, T, n):
    # Gives the sum corresponding to the T most probable bit patterns for the n llr values
    
    bitPatternSum = 1 # Start with the value corresponding to the zeros only pattern
    
    # A matrix with the candidates for the next most probable vector, the bit to be changed for the next vector
    C = np.zeros ((T, n + 1), dtype = 'int')
    
    # The llr sum for the next vector
    # C_llrsum = np.zeros((T, 1))
    C_llrsum = np.zeros(T)
    C_llrsum [0] = llr[0] # The corresponding llr sum is llr[0]
    
    for m in xrange(1, T):
        
        bitPatternSum = bitPatternSum + exp(-C_llrsum[0]) # Add the term corresponding to the next bit pattern to add
        
        index = C[0, -1] # The index for the bit position that should be 1 in the next candidate vector
        vector = C[0, 0:n].copy() # The new candidate vector
        vector[index] = 1
        # Modify the parent node and add the child node in C
        
        if index < n - 1: # Add the parent and child node to list of candidates
            # Child node
            C[m, 0:n] = vector # Adds the new vector to the set of candidates
            C[m, -1] = index + 1 # Sets its next index
            C_llrsum[m] = C_llrsum[0] + llr[index + 1] # Sets its llr value

            # Parent node
            C[0, -1] = index + 1 # Updates next index for parent node
            C_llrsum[0] = C_llrsum[0] + llr[index + 1] - llr[index] # Updates llr sum for parent node
        else:
            C_llrsum[0] = sys.maxint # Makes it impossible to choose a vector with an index outside of the vector size
            C_llrsum[m] = sys.maxint # Also make it impossible to pick the child node!
        
        # Sort the list of candidates according to the llr values
        Illr = np.argsort(C_llrsum[0:m].squeeze())
        C_llrsum[0:m] = C_llrsum[Illr]
        C[0:m, :] = C[Illr, :]
    
    return bitPatternSum
    
def bitPatternListSumImproved(llr, T, n):
    # Gives the sum corresponding to the T most probable bit patterns for the n llr values
    
    bitPatternSum = 1 # Start with the value corresponding to the zeros only pattern
    
    C = [] # The priority queue of candidates
    
    llrSum = llr[0]
    index = 0
    
    for m in xrange(1, T):
        bitPatternSum = bitPatternSum + exp(-llrSum) # Add the llr sum of the current bit pattern
        
        if index < n - 1: # Add the parent and child node to list of candidates if the index of the current candidate is less than n - 1
            # Push the child node
            hq.heappush(C, (llrSum + llr[index + 1], index + 1))

            # Push the parent node and pop
            (llrSum, index) = hq.heappushpop(C, (llrSum + llr[index + 1] - llr[index], index + 1))
        else: # Otherwise just pop the top candidate
            (llrSum, index) = hq.heappop(C)
    
    return bitPatternSum

def probabilityEstimationSoftStern(n, k, sigma, l, T, numOfSims):
    # Estimates the probability of Soft Stern succeeding
    # The problem parameters are n, k and sigma
    # The algorithm parameters are l and T
    # We run the simulation numOfSims times
    
    G = getRandomGeneratorMatrix(k, n) # Gets a random generator matrix
    
    gaussian = RealDistribution('gaussian', sigma) # The normal distribution
#    gaussian.set_seed(0.0) # # To make results reproducible, might want to change this
    
    Ptotal = 0
    
    for j in range(numOfSims):
    
        print j
    
        u = np.random.randint(2, size = k)
        c = np.array(np.mod(u*G, 2)).squeeze()
        v = 1 - 2*c.astype('float') # Transforms the encoding
        r = v + [gaussian.get_random_element() for i in xrange(n)] # Generate a random r vector

        I = np.argsort([-abs(r)]) # Not optimal to negate the vector
        r = r[I].squeeze() # Sorts the r values according to their reliability values

        I2 = np.random.permutation(range(k + l)) # Random permutation of the k + l most reliable positions
        r = abs(r[I2].squeeze()) # Random permutation of the k + l most reliable positions, don't care about the sign of the positions

        IS1 = np.argsort(r[0:(k + l)//2])
        IS2 = np.argsort(r[(k + l)//2:k + l])

        IS2 = IS2 + (k + l)//2

        r[0:(k + l)//2] = r[IS1]
        r[(k + l)//2:k + l] = r[IS2]

        llr = 2*r/sigma**2
        
        
        # sum1 = bitPatternListSum(llr[0:floor((k + l)/2)], T, floor((k + l)/2))
        # sum2 = bitPatternListSum(llr[floor((k + l)/2):k + l], T, ceil((k + l)/2))
        
        sum1 = bitPatternListSumImproved(llr[0:floor((k + l)/2)], T, floor((k + l)/2))
        sum2 = bitPatternListSumImproved(llr[floor((k + l)/2):k + l], T, ceil((k + l)/2))       
        
        # Calculate Q_l
        # Ql = np.prod([1 - 0.5**i for i in range(l + 1, l + 100)]) # Probability of finding an invertible submatrix
        
        # Calculate P1 and P2
        P1 = np.prod(1/(1 + exp(-abs(llr[0:floor((k + l)/2)]))))
        P2 = np.prod(1/(1 + exp(-abs(llr[floor((k + l)/2):k + l]))))

        Ptotal = Ptotal + P1*sum1 * P2*sum2
        # Ptotal = Ptotal +  Ql * P1*sum1 * P2*sum2

    PA = Ptotal/numOfSims # Estimate probability of success by the average
    
    return PA
    
def probabilityEstimationBasicOSD(n, k, sigma, T, numOfSims):
    # Estimates the probability of basic OSD
    # The problem parameters are n, k and sigma
    # The algorithm parameter is T
    # We run the simulation numOfSims times
    
    G = getRandomGeneratorMatrix(k, n) # Gets a random generator matrix
    
    gaussian = RealDistribution('gaussian', sigma) # The normal distribution
#    gaussian.set_seed(0.0) # # To make results reproducible, might want to change this
    
    Ptotal = 0
    
    for j in range(numOfSims):
    
#        print j
    
        u = np.random.randint(2, size = k)
        c = np.array(np.mod(u*G, 2)).squeeze()
        v = 1 - 2*c.astype('float') # Transforms the encoding        
        
        r = v + [gaussian.get_random_element() for i in xrange(n)] # Generate a random r vector

        I = np.argsort([-abs(r)]) # Not optimal to negate the vector
        r = abs(r[I].squeeze()) # Sorts the r values according to their reliability values

        llr = 2*r/sigma**2      
        
        # sum1 = bitPatternListSum(llr[0:floor((k + l)/2)], T, floor((k + l)/2))
        # sum2 = bitPatternListSum(llr[floor((k + l)/2):k + l], T, ceil((k + l)/2))
        
        sum1 = bitPatternListSumImproved(llr[0:k], T, k)
        
        # Calculate Q_l
        # Ql = np.prod([1 - 0.5**i for i in range(1, 100)]) # Probability that the first k columns form an invertible matrix
        
        # Calculate P1 and P2
        P1 = np.prod(1/(1 + exp(-abs(llr[0:k]))))

        Ptotal = Ptotal + P1*sum1
        # Ptotal = Ptotal + Ql*P1*sum1

    PA = Ptotal/numOfSims # Estimate probability of success by the average
    
    return PA


def theCryptoSimulation():
    K = [128, 256, 512, 1024, 2048]
    SigmaHigh = [1, 1, 1, 1, 1]
    SigmaLow = [0.65, 0.65, 0.65, 0.65, 0.65]
    numOfSims = 12
    numOfSigmaValues = 15
    # l = 20
    l = 10
    T = 2**l
    f = open("results/softStern_l_20.txt", "w+")
    
    for i in range(5):
        k = K[i]
        n = 2*k
        print("Starts the k = " + str(k) + " case!")
        sigmaLow = SigmaLow[i]
        sigmaHigh = SigmaHigh[i]
        
        Sigma = np.linspace(sigmaLow, sigmaHigh, numOfSigmaValues)
        print("sigma P_{soft}(A) P_{basic}(A)")
        
        for sigma in Sigma:
            PA = probabilityEstimationSoftStern(n, k, sigma, l, T, numOfSims)
            PABasic = probabilityEstimationBasicOSD(n, k, sigma, 2*T, numOfSims) # Should we let Basic OSD use 2*T bit patterns?
            print(str(sigma) + " " + str(PA) + " " + str(PABasic))
    
def theCodingSimulation():
    K = [64, 128, 256, 512, 1024]
    SigmaHigh = [0.65, 0.65, 0.65, 0.65, 0.65]
    SigmaLow = [0.4, 0.4, 0.4, 0.4, 0.4]
    numOfSims = 1000
    numOfSigmaValues = 11
    # l = 20
    l = 10
    T = 2**l
    
    for i in range(5):
        k = K[i]
        n = 2*k
        print("Starts the k = " + str(k) + " case!")
        sigmaLow = SigmaLow[i]
        sigmaHigh = SigmaHigh[i]
        
        Sigma = np.linspace(sigmaLow, sigmaHigh, numOfSigmaValues)
        print("sigma log2(1 - P_{soft}(A)) log2(1 - P_{basic}(A))")
        
        for sigma in Sigma:
            PA = probabilityEstimationSoftStern(n, k, sigma, l, T, numOfSims)
            PABasic = probabilityEstimationBasicOSD(n, k, sigma, 2*T, numOfSims) # Should we let Basic OSD use 2*T bit patterns?
            print(str(sigma) + " " + str(np.log2(1 -PA)) + " " + str(np.log2(1 - PABasic)))    

##### Functions for simulating the P(A) values ends here #####




##### Main part begins here #####
			
#np.random.seed(0) # To make results reproducible
# theCryptoSimulation()
# theCodingSimulation()

# print "Done!"

k = 3601
n = 2*k
sigma = 0.66
numOfSims = 100

l = 23
T = 2**l


start = timer() # Starts measuring the elapsed time


PA = probabilityEstimationSoftStern(n, k, sigma, l, T, numOfSims)
# PABasic = probabilityEstimationBasicOSD(n, k, sigma, 2*T, numOfSims) # Should we let Basic OSD use 2*T bit patterns?

end = timer() # Ends measuring the elapsed time
print "Total time spent: " + str(end - start) + " s."

print "Probability when using soft Stern: " + str(PA)
# print "Probability when using basic OSD: " + str(PABasic)

# print "Log2 of probability of failing when using soft Stern: " + str(np.log2(1 - PA))
# print "Log2 of probability of failing when using basic OSD: " + str(np.log2(1 - PABasic))


##### Main part ends here #####