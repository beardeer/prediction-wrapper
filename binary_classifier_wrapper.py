from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import numpy as np
import pandas as pd

from model_wrapper import ModelWrapper

class KfoldBinaryClassifierWrapper(ModelWrapper):

	def __init__(self, data_frame, feature_names, label_name, categorical_feature_names = None, k = 5):
		ModelWrapper.__init__(self, data_frame, feature_names, label_name, categorical_feature_names)

		self.cv = None
		self.k = k
		self._generate_split_index()

	def _generate_split_index(self):
		self.cv = KFold(n_splits = self.k, shuffle = True)

	def _split_data(self, train_idx, test_idx):
		x_train = (self.data_frame[self.feature_names].iloc[train_idx,:])
		y_train = self.data_frame[self.label_name].iloc[train_idx]
		x_test = (self.data_frame[self.feature_names].iloc[test_idx,:])
		y_test = self.data_frame[self.label_name].iloc[test_idx]

		return x_train, y_train, x_test, y_test

	def _transform_categorical_featurs(self):

		le = LabelEncoder()
		for name in self.categorical_feature_names:
			self.data_frame[name] = le.fit_transform(self.data_frame[name])

	def _onehot_categorical_featurs(self, train_data, test_data):

		if self.categorical_feature_names == None:
			return train_data, test_data

		feature_idxs = [self.data_frame.columns.get_loc(name) for name in self.categorical_feature_names]
		encoder = OneHotEncoder(categorical_features = feature_idxs)
		encoder.fit(np.vstack((train_data, test_data)))
		train_data = encoder.transform(train_data)
		test_data = encoder.transform(test_data)
		
		return train_data, test_data

	def run(self):

		results = pd.DataFrame(index = [i for i in range(len(self.data_frame))], columns = ['label', 'pred_prob', 'pred_label'])

		self._transform_categorical_featurs()

		for train_idx, test_idx in self.cv.split(self.data_frame):
			x_train, y_train, x_test, y_test = self._split_data(train_idx, test_idx)
			x_train, x_test = self._onehot_categorical_featurs(x_train, x_test)

        	self.model.fit(x_train, y_train)

        	y_pred_p = self.model.predict_proba(x_test)[:, 1]
        	y_pred_l = self.model.predict(x_test)

        	results['label'].iloc[test_idx] = y_test
        	results['pred_prob'].iloc[test_idx] = y_pred_p
        	results['pred_label'].iloc[test_idx] = y_pred_l
		
		return results



