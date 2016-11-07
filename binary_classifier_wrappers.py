"""The implementation of classifier wrappers
"""

# Author: Xiaolu Xiong <beardeer@gmail.com>

from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import numpy as np
import pandas as pd

from .model_wrapper import ModelWrapper
from .pred_results import BinaryPredResult

class KfoldBinaryClassifierWrapper(ModelWrapper):
    """The class runs k-fold cross-validation on a sklearn classifier model.

    Attributes
    ----------
    k : int
        The number of k folds in cross-validation
    kfold : sklearn KFold
        K-Folds cross-validator for a given k
    """
    def __init__(self, data_frame, label_name, feature_names, 
    	categorical_feature_names = [], k = 5):
        ModelWrapper.__init__(self, data_frame, label_name, feature_names, categorical_feature_names)

        self.k = k
        self._kfold = None
        self._generate_kfold()

    def _generate_kfold(self):
        """Generate a K-Folds cross-validator for a given k.

        Returns
        -------
        None
        """
        self._kfold = KFold(n_splits = self.k, shuffle = True)

    def _split_data(self, train_idx, test_idx):
        """Split the data_frame by training index and testing index. 

        Parameters
        ----------
        train_idx : list
            A list of traning indexes
        test_idx : list
            A list of testing indexes

        Returns
        -------
        DataFrames
            DataFrames that used in current fold of cross validation
        """
        x_train = (self.data_frame[self.feature_names].iloc[train_idx,:])
        y_train = self.data_frame[self.label_name].iloc[train_idx]
        x_test = (self.data_frame[self.feature_names].iloc[test_idx,:])
        y_test = self.data_frame[self.label_name].iloc[test_idx]
        return x_train, y_train, x_test, y_test

    def _transform_categorical_featurs(self):
        """Utilize the sklearn LabelEncoder to encode categorical features.

        This is necessary for using categorical values that are represented in strings. 

        Returns
        -------
        None
        """
        le = LabelEncoder()
        # Transform each categorical feature with value between 0 and n_classes-1
        for name in self.categorical_feature_names:
            self.data_frame[name] = le.fit_transform(self.data_frame[name])

    def _onehot_categorical_featurs(self, train_data, test_data):
        """Utilize the sklearn OneHotEncoder to encode categorical features.

        In order to feed categorical features to sklearn model, we need to convert
        them to one-hot encoding. 

        There are other different ways of using categorical features, such as using
        pandas.get_dummies, but get_dummies will create additional features in
        data_frame, which I perfer not to that.  

        Parameters
        ----------
        train_data : DataFrame
            Traning data
        test_data : DataFrame
            Testing data

        Returns
        -------
        numpy matrix
            Two numpy matrices with one-hot encoded features
        """
        if self.categorical_feature_names == []:
            return train_data, test_data
        # Select indexs of categorical features, and encode, then transform them
        feature_idxs = [self.data_frame.columns.get_loc(name) for name in self.categorical_feature_names]
        encoder = OneHotEncoder(categorical_features = feature_idxs)
        # Need to fit on every values from both training and testing
        encoder.fit(np.vstack((train_data, test_data)))
        train_data = encoder.transform(train_data)
        test_data = encoder.transform(test_data)

        return train_data, test_data

    def run(self):
        """Run classifier model with k-fold cross-validation

        Returns
        -------
        PredResult
            Model-generated Prediction results
        """
        results = BinaryPredResults(len(self.data_frame))
        self._transform_categorical_featurs()

        # Run k-fold cross-validation
        for train_idx, test_idx in self._kfold.split(self.data_frame):
            x_train, y_train, x_test, y_test = self._split_data(train_idx, test_idx)
            x_train, x_test = self._onehot_categorical_featurs(x_train, x_test)

            # Training
            self.model.fit(x_train, y_train)

            # Testing and generating predictions
            y_pred_p = self.model.predict_proba(x_test)[:, 1]
            y_pred_l = self.model.predict(x_test)

            results.set_col(y_test, 'label', test_idx)
            results.set_col(y_pred_p, 'pred_prob', test_idx)
            results.set_col(y_pred_l, 'pred_label', test_idx)

        return results



