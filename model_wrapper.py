from abc import ABCMeta, abstractmethod

class ModelWrapper(object):

    __metaclass__ = ABCMeta

    def __init__(self, data_frame, label_name, feature_names, categorical_feature_names = []):
        self.data_frame = data_frame
        self.label_name = label_name
        self.feature_names = feature_names
        self.categorical_feature_names = categorical_feature_names
        self.model = None

    def set_model(self, model):
        self.model = model

    @abstractmethod
    def run(self):

        raise NotImplementedError()
        

