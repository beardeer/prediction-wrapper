"""Usefule performance metric 
"""

# Author: Xiaolu Xiong <beardeer@gmail.com>

from abc import ABCMeta, abstractmethod
from scipy import stats
from sklearn import metrics

class MetricWrapper(object):
	"""The abstract class to build performance metrics
	"""
	__metaclass__ = ABCMeta
	
	@classmethod
	@abstractmethod
	def measure(cls, label, prediction):
		"""The abstract class method to measure the performance of predictions

		Parameters
		----------
		label : array
		    Lable data
		prediction : array
		    Prediction data

        Raises
        ------
        NotImplementedError
		"""
		raise NotImplementedError()


class RSquare(MetricWrapper):
	"""The implementation of simple coefficient of determination.

	It is the square of pearson correlation.  
	"""
	@classmethod
	def measure(cls, label, prediction):
		"""Measure the simple coefficient of determination.

		Parameters
		----------
		label : array
		    Lable data
		prediction : array
		    Prediction data

		Returns
		-------
		float
		    The square of pearson correlation between label and prediction
		"""
		r, _ = stats.pearsonr(label, prediction)
		return r**2

class AUC(MetricWrapper):
	"""The implementation of Area under the ROC curve. 
	"""
	@classmethod
	def measure(cls, label, prediction):
		"""Measure the AUC.

		Parameters
		----------
		label : array
		    Lable data
		prediction : array
		    Prediction data

		Returns
		-------
		float
		    The AUC value between label and prediction
		"""
		return metrics.roc_auc_score(label, prediction)


class RMSE(MetricWrapper):
	"""The implementation of root-mean-square error. 
	"""
	@classmethod
	def measure(cls, label, prediction):
		"""Measure the RMSE

		Parameters
		----------
		label : array
		    Lable data
		prediction : array
		    Prediction data

		Returns
		-------
		float
		    The RMSE value of lable and prediction
		"""
		return metrics.mean_squared_error(label, prediction)**0.5

