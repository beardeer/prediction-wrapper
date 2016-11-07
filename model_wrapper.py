"""
The abstrct class for builidng prediction model wrappers
"""

# Author: Xiaolu Xiong <beardeer@gmail.com>


from abc import ABCMeta, abstractmethod

class ModelWrapper(object):
    """The abstrct class for builidng prediction model wrappers. 

    Attributes
    ----------
    categorical_feature_names : list
        A list contains categorical feature names, these names must appear in feature_name
    data_frame : pandas.DataFrame
        The input data for building models
    feature_names : list
        A list of input feature names, these names must appear in data_frame's column names
    label_name : str
        A column name in data_frame to local the label
    model : sklearn BaseEstimator
        A sklearn prediction model
    """
    __metaclass__ = ABCMeta

    def __init__(self, data_frame, label_name, 
        feature_names, categorical_feature_names = []):

        self.data_frame = data_frame
        self.label_name = label_name
        self.feature_names = feature_names
        self.categorical_feature_names = categorical_feature_names
        self.model = None

    def set_model(self, model):
        """set a prediction model

        Parameters
        ----------
        model : sklearn BaseEstimator
            A sklearn prediction model used in current model wrapper

        Returns
        -------
        None
        """
        self.model = model

    @abstractmethod
    def run(self):
        """Abstract method for running model with data_frame

        Raises
        ------
        NotImplementedError
            Description
        """
        raise NotImplementedError()
        

