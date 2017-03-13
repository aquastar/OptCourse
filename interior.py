# training_set =
# 1: (1,1) (2,2) (2,0) (2,1)
# 2: (0,0) (1,0) (0,1) (-1,-1)

import svmcmpl
import time
from cvxopt import matrix

X = matrix([1.0, 2.0, 2.0, 2.0, 0.0, 1.0, 0.0, -1.0, 1.0, 2.0, 0.0, 1.0, 0.0, 0.0, 1.0, -1.0], (8, 2))
d = matrix([1, 1, 1, 1, -1, -1, -1, -1])
gamma = 2.0
kernel = 'linear'
sigma = 1.0

sol1 = svmcmpl.softmargin(X, d, gamma, kernel, sigma)
