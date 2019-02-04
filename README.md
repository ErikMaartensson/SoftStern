# SoftStern
An implementation of the algorithm described in "Some Cryptanalytic and Coding-Theoretic Applications of a Soft Stern Algorithm". It covers the single-iteration algorithm as described in Section 3 of the paper. It is not that optimized for speed at the moment. 

softStern.py contains the Soft Stern algorithm. It also contains code for simulating P(A) when using the Soft Stern algorithm as described in the paper, or when using a basic OSD algorithm.

BMA.py and BMA_crypto.py contain code for simulating P(A) when using the box-and-match algorithm. HStern.py and HStern_crypto.py contain code for simulating P(A) when using the hard-decision Stern algorithm.

The code in general is at a working state and needs to be a bit polished. It can be used as a starting-point to write an efficient implementation of the soft Stern algorithm.
