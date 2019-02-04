# -*- mode: sage-shell:sage -*-
# -*- coding: utf-8 -*-

import itertools as it
import numpy as np
import scipy.special as ss
from timeit import default_timer as timer

# GENERATOR AND PARITY-CHECK MATRICES BEGIN

def getRandomGeneratorMatrix(K, N):
    I = np.eye(K, dtype = 'int')
    P = np.matrix(np.random.randint(2, size = (K, N - K)), dtype = 'int')
    G = np.bmat([I, P])
    return G

    # return np.matrix(np.random.randint(2, size = (K, N)))


# HYBRID LDPC PARITY-CHECK MATRICES BEGIN
def encrypt(u, G, N, sigma):
    # Encodes the message
    c = np.array(np.mod(u*G, 2)).squeeze()
    v = 1 - 2*c.astype('float') # Transforms the encoding

    # Adds random noice to the vector v
    gaussian = RealDistribution('gaussian', sigma)
    gaussian.set_seed(0.0) # # To make results reproducible, might want to change this
    r = v + [gaussian.get_random_element() for i in xrange(N)]
    return r
    
def invOrder(v, I, shift = 0):
    v_copy = v.copy()
    for i in xrange(len(I)):
        v[I[i]] = v_copy[i + shift]

def decode_hard(r,N=N):
    '''
    bit-wise hard decision decoding. Assume that the messages are all-0, i.e., 1 after mapping (-1)^c_i.
    '''
    h = [0 for i in range(N)]
    for i in range(N):
        if r[i] < 0:
            h[i] = 1
    return h

def maxLSize(T, n, l_max):
    '''
    Return the maximum value of l such that \sum^l binomial(n,j) \leq T
    '''

    A = [0 for i in range(l_max)]
    A[0] = 1
    i = 1
    while A[i-1] <= T:
        A[i] = A[i-1] + ss.binom(n,i)
        i = i+1

    return i-2, T-A[i-2]



# paramters

np.random.seed(0) # To make results reproducible


# Estimate the probability for pure stern:

# sim = 1000
# l_max = 20

# k = 96
# n
n = 2*k
# N = [96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140, 144, 148, 152, 156, 160, 164, 168, 172, 176, 180, 184, 188, 192]


# Sigma = np.linspace(sigma0, sigma1, 2)
N = n





# Runs estimations for multiple values of sigma and multiple values of l
# Just theoretical estimation for






for sigma in Sigma:
    log_com = 100000
    l_best = 0
    gaussian = RealDistribution('gaussian', sigma)
    gaussian.set_seed(0.0) # # To make results reproducible, might want to change this
    set_random_seed(0)
    for l in range(l_start, l_max+1):

        # G = getRandomGeneratorMatrix(k, n)
        T = 2**l
        # successes


        # Calculate Q_l
        Ql = np.prod([1 - 0.5**i for i in range(l + 1, l + 100)]) # Probability of finding an invertible submatrix

        # the size of the first part
        size = floor((k + l)/2)
        weight1, num_rem1 = maxLSize(T, size, l_max)
        weight2,  num_rem2 = maxLSize(T, k+l-size, l_max)
        # print  weight1, num_rem1
        # print  weight2, num_rem2

        PA_hstern = 0.0 # estimated success probability for the hard-stern algorithm.
        # PA_MBA = 0.0 # estimated success probability for the box-match algorithm.

        for j in range(sim):
            # print j
            # Adds random noice to the vector v
            w = [gaussian.get_random_element() for i in xrange(N)]
            r = [w[i]+1.0 for i in xrange(N)]
            # c_hard = decode_hard(r,N)
            # print c_hard
            llr = 2*r/sigma**2
            # llr = [2*r[i]/sigma**2 for i in xrange(N)]



            # Calculate P1 and P2
            P1_hstern = np.prod(1/(1 + exp(-abs(llr[0:floor((k + l)/2)]))))
            P2_hstern = np.prod(1/(1 + exp(-abs(llr[floor((k + l)/2):k + l]))))
            # print P1_hstern, P2_hstern

            sum1_hstern = 0.0


            for t in range(0,weight1 +1):
                if t == 0:
                    sum1_hstern += 1.0
                    continue
                for pattern in it.combinations(range(size), t):
                    log_sum = 0
                    for itera in range(t):
                        log_sum -= llr[pattern[itera]]
                    sum1_hstern += exp(log_sum)

            num_curr = num_rem1

            for pattern in it.combinations(range(size), weight1 +1):
                log_sum = 0
                for itera in range(weight1+1):
                    log_sum -= llr[pattern[itera]]
                sum1_hstern += exp(log_sum)
                num_curr -= 1
                if num_curr <= 0:
                    # print num_curr
                    break


            sum2_hstern = 0.0

            for t in range(0,weight2 +1):
                if t == 0:
                    sum2_hstern += 1.0
                    continue
                for pattern in it.combinations(range(size,k+l), t):
                    log_sum = 0
                    for itera in range(t):
                        log_sum -= llr[pattern[itera]]
                    sum2_hstern += exp(log_sum)

            num_curr = num_rem2

            for pattern in it.combinations(range(size,k+l), weight2 +1):
                log_sum = 0
                for itera in range(weight1+1):
                    log_sum -= llr[pattern[itera]]
                sum2_hstern += exp(log_sum)
                num_curr -= 1
                if num_curr <= 0:
                    # print num_curr
                    break
            PA_hstern_temp = min(1, P1_hstern*sum1_hstern * P2_hstern*sum2_hstern)
            PA_hstern += PA_hstern_temp
            # The probability of hard stern to find an invertible matrix is clost to 1.

            # I0 = np.argsort([-abs(r[i]) for i in range(N)])
            # # print I
            # # print r[I[0]], r[I[1]], r[I[2]], r[I[4]]
            # I_MBA = np.random.permutation(range(k + l)) # Random permutation of the k + l most reliable positions
            # # print I_MBA
            # I = [I0[I_MBA[i]] for i in range(k+l)]

            # # Calculate P1 and P2
            # P1_MBA = np.prod(1/(1 + exp(-abs(llr[I[0:floor((k + l)/2)]]))))
            # P2_MBA = np.prod(1/(1 + exp(-abs(llr[I0[I_MBA[floor((k + l)/2):k + l]]]))))
            # # print P1_MBA, P2_MBA


            # sum1_MBA = 0.0

            # for t in range(weight1 +1):
            #     for pattern in it.combinations(I[0:size], t):
            #         log_sum = 0
            #         for itera in range(t):
            #             log_sum -= llr[pattern[itera]]
            #         sum1_MBA += exp(log_sum)


            # num_curr = num_rem1

            # for pattern in it.combinations(I[0:size], weight1 +1):
            #     log_sum = 0
            #     for itera in range(weight1+1):
            #         log_sum -= llr[pattern[itera]]
            #     sum1_hstern += exp(log_sum)
            #     num_curr -= 1
            #     if num_curr <= 0:
            #         # print num_curr
            #         break


            # sum2_MBA = 0.0

            # for t in range(weight2 +1):
            #     for pattern in it.combinations(I[size:k+l], t):
            #         log_sum = 0
            #         for itera in range(t):
            #             log_sum -= llr[pattern[itera]]
            #         sum2_MBA += exp(log_sum)


            # num_curr = num_rem2

            # for pattern in it.combinations(I[size:k+l], weight1 +1):
            #     log_sum = 0
            #     for itera in range(weight1+1):
            #         log_sum -= llr[pattern[itera]]
            #     sum2_hstern += exp(log_sum)
            #     num_curr -= 1
            #     if num_curr <= 0:
            #         # print num_curr
            #         break

            # PA_MBA +=  Ql * P1_MBA*sum1_MBA * P2_MBA*sum2_MBA

        print "sigma = " + str(sigma)
        print "l = " + str(l)
        # PA_MBA = PA_MBA / float(sim)
        PA_hstern /= float(sim)

        print(PA_hstern)
        # print(PA_MBA)S

        print np.log2(PA_hstern)

        log_com_temp = np.log2(T/PA_hstern)
        if log_com_temp < log_com:
            log_com = log_com_temp
            l_best = l
    print "the best log2 complexity is", log_com
    print "the best l is", l_best