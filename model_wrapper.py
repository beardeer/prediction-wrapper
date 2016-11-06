from abc import ABCMeta, abstractmethod

class ModelWrapper(object):

    __metaclass__ = ABCMeta

    def __init__(self, data_frame, feature_names, label_name, categorical_feature_names = None):
        self.data_frame = data_frame
        self.feature_names = feature_names
        self.label_name = label_name
        self.categorical_feature_names = categorical_feature_names
        self.model = None

    def set_up_model(self, model):
        self.model = model

    @abstractmethod
    def run(self):

        raise NotImplementedError()
        

