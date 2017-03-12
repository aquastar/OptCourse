# training_set =
# 1: (1,1) (2,2) (2,0) (2,1)
# 2: (0,0) (1,0) (0,1) (-1,-1)

import cvxopt, svmcmpl
from cvxopt import matrix

X = matrix([1.0, 2.0, 2.0, 2.0, 0.0, 1.0, 0.0, -1.0, 1.0, 2.0, 0.0, 1.0, 0.0, 0.0, 1.0, -1.0], (8, 2))
d = matrix([1, 1, 1, 1, -1, -1, -1, -1])
print X, d
gamma = 2.0
kernel = 'rbf'
sigma = 1.0
# width = 20
sol1 = svmcmpl.softmargin(X, d, gamma, kernel, sigma)
# sol2 = svmcmpl.softmargin_appr(X, d, gamma, width, kernel, sigma)
