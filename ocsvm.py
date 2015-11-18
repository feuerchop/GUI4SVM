__author__='morgan'

import numpy as np
from sklearn.svm import OneClassSVM


class ocsvm(OneClassSVM):
    """
    One-class SVM class subclassed from OneClassSVM defined in sklearn
    Because the super predict function is not what we expected. We need
    to rewrite it to meet our needs.
    """

    # TODO: Add more properties to OCSVM
    def __init__(self, *args, **kwargs):
        super(ocsvm, self).__init__(*args, **kwargs)
        self.eps = 1e-4
        self.fval = None
        self.sv_ind = []
        self.bsv_ind = []

    def fit(self, *args, **kwargs):
        super(ocsvm, self).fit(*args, **kwargs)
        self.sv_ind = np.where(self.dual_coef_.ravel() < 1-self.eps)[0]
        self.bsv_ind = np.setdiff1d(self.support_, self.sv_ind)

    def predict_y(self,X):
        """
        :rtype : ndarray
        :param X: training data X
        :return: predicted labels, +1 for outlier, -1 for normal
        """
        dec=self.decision_function(X)
        # find the nearest BSV
        threshold = self.decision_function(self.support_vectors_[self.sv_ind, :]).min()
        yc = dec.ravel()
        pos = yc < threshold
        neg = yc >= threshold
        yc[neg] = -1        # negative samples refer to normal
        yc[pos] = 1          # positive samples refer to outlier
        return yc


