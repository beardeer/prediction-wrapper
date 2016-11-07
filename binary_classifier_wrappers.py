"""Summary
"""
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import numpy as np
import pandas as pd

from .model_wrapper import ModelWrapper
from .pred_results import BinaryPredResult

class KfoldBinaryClassifierWrapper(ModelWrapper):
    """Summary

    Attributes
    ----------
    k : TYPE
        Description
    kfold : TYPE
        Description
    """
    def __init__(self, data_frame, label_name, feature_names, categorical_feature_names = [], k = 5):
        """Summary

        Parameters
        ----------
        data_frame : TYPE
            Description
        label_name : TYPE
            Description
        feature_names : TYPE
            Description
        categorical_feature_names : list, optional
            Description
        k : int, optional
            Description
        """
        ModelWrapper.__init__(self, data_frame, label_name, feature_names, categorical_feature_names)

        self.k = k
        self.kfold = None
        self._generate_split_index()

    def _generate_split_index(self):
        """Summary

        Returns
        -------
        TYPE
            Description
        """
        self.kfold = KFold(n_splits = self.k, shuffle = True)

    def _split_data(self, train_idx, test_idx):
        """Summary

        Parameters
        ----------
        train_idx : TYPE
            Description
        test_idx : TYPE
            Description

        Returns
        -------
        TYPE
            Description
        """
        x_train = (self.data_frame[self.feature_names].iloc[train_idx,:])
        y_train = self.data_frame[self.label_name].iloc[train_idx]
        x_test = (self.data_frame[self.feature_names].iloc[test_idx,:])
        y_test = self.data_frame[self.label_name].iloc[test_idx]
        return x_train, y_train, x_test, y_test

    def _transform_categorical_featurs(self):
        """Summary

        Returns
        -------
        TYPE
            Description
        """
        le = LabelEncoder()
        for name in self.categorical_feature_names:
            self.data_frame[name] = le.fit_transform(self.data_frame[name])

    def _onehot_categorical_featurs(self, train_data, test_data):
        """Summary

        Parameters
        ----------
        train_data : TYPE
            Description
        test_data : TYPE
            Description

        Returns
        -------
        TYPE
            Description
        """
        if self.categorical_feature_names == []:
            return train_data, test_data

        feature_idxs = [self.data_frame.columns.get_loc(name) for name in self.categorical_feature_names]
        encoder = OneHotEncoder(categorical_features = feature_idxs)
        encoder.fit(np.vstack((train_data, test_data)))
        train_data = encoder.transform(train_data)
        test_data = encoder.transform(test_data)

        return train_data, test_data

    def run(self):
        """Summary

        Returns
        -------
        TYPE
            Description
        """
        result = BinaryPredResult(len(self.data_frame))
        self._transform_categorical_featurs()

        for train_idx, test_idx in self.kfold.split(self.data_frame):
            x_train, y_train, x_test, y_test = self._split_data(train_idx, test_idx)
            x_train, x_test = self._onehot_categorical_featurs(x_train, x_test)

            self.model.fit(x_train, y_train)

            y_pred_p = self.model.predict_proba(x_test)[:, 1]
            y_pred_l = self.model.predict(x_test)

            result.set_col(y_test, 'label', test_idx)
            result.set_col(y_pred_p, 'pred_prob', test_idx)
            result.set_col(y_pred_l, 'pred_label', test_idx)

        return result



