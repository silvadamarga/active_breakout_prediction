#!/usr/bin/env python3
"""
v6 - Cross-Validation
Contains the Purged K-Fold class for time-series validation.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

class PurgedKFold(KFold):
    """
    Implements Purged K-Fold Cross-Validation.
    - Purges training samples that overlap with test samples.
    - Embargoes training samples that are too close to the test set.
    """
    def __init__(self, n_splits=5, t1=None, pctEmbargo=0.01):
        super(PurgedKFold, self).__init__(n_splits, shuffle=False, random_state=None)
        self.t1 = t1 # A pd.Series of label end-times, indexed by observation
        self.pctEmbargo = pctEmbargo

    def split(self, X, y=None, groups=None):
        if X.shape[0] == 0:
            raise ValueError("Input X cannot be empty")

        indices = np.arange(X.shape[0])
        mbrg = int(X.shape[0] * self.pctEmbargo)
        test_starts = [(i[0], i[-1] + 1) for i in np.array_split(indices, self.n_splits)]

        for i, j in test_starts:
            if self.t1 is None:
                # Standard K-Fold if no time series info
                train_indices = np.concatenate([indices[:i], indices[j:]])
                test_indices = indices[i:j]
            else:
                if i >= len(self.t1): continue 
                t0 = self.t1.iloc[i] # Start of test set time
                test_indices = indices[i:j]
                
                if j < len(self.t1):
                    t1_end = self.t1.iloc[j-1] # End of test set time (inclusive index)
                else:
                    t1_end = self.t1.iloc[-1] 

                # Embargo: Remove samples immediately following the test set
                train_indices_embargoed = np.concatenate([
                    indices[:i], # Indices before test start
                    indices[j + mbrg:] # Indices after test end + embargo gap
                ])

                # Purge: Remove training samples whose labels overlap with the test set
                if train_indices_embargoed.size > 0:
                    train_t1 = self.t1.iloc[train_indices_embargoed]
                    train_indices = train_indices_embargoed[
                        (train_t1 < t0) | (train_t1 > t1_end)
                    ]
                else:
                    train_indices = train_indices_embargoed

            yield train_indices, test_indices
