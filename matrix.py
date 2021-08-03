#!/usr/bin/env python3
"""
Contains the Identity class that represents the indentity matrix of size n.

Copy/pasted from the matrix.py file of the original repository:
https://github.com/usnistgov/PrivacyEngCollabSpace/tree/master/tools
de-identification/Differential-Privacy-Synthetic-Data-Challenge-Algorithms/
rmckenna
"""

import numpy as np
from scipy.sparse.linalg import LinearOperator


class Identity(LinearOperator):
    """Represents the indentity matrix of size n."""

    def __init__(self, n, dtype=np.float64):
        """Initialize the identity matrix of size n.

        Args:
            n: The size of the identity matrix.
            dtype: The data type.
        """
        self.shape = (n, n)
        self.dtype = dtype

    def _matmat(self, V):
        return V

    def _transpose(self):
        return self

    def _adjoint(self):
        return self
