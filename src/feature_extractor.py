import os
import numpy as np 
from keras.models import load_model, Model
from .build_model import ArcFaceLossLayer, dummy_loss
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

class FeatureExtractor():
    def __init__(self, model_path, num_classes):
        try:
            arcface_layer = ArcFaceLossLayer(num_classes=num_classes)
            self.model = load_model(model_path, custom_objects={'ArcFaceLossLayer':arcface_layer, 'dummy_loss':dummy_loss})
            self.num_classes = num_classes
            self.__get_infer_model()
            print('** Sussesed to load model from {}'.format(model_path))
        except Exception as e:
            print('** Failed to load model from {}'.format(model_path))
            print(e)
    
    def __get_infer_model(self):
        self.infer_model = Model(inputs=self.model.input, outputs=self.model.get_layer('embeddings').output)

    def __normalize(self, img):
        return np.multiply(np.subtract(img, 127.5), 1 / 128)

    def infer(self, faces):
        normalized = [self.__normalize(f) for f in faces]
        normalized = np.array(normalized)
        if len(normalized.shape) < 4:
            normalized = np.expand_dims(normalized, axis=0) 
        dummy_y = np.zeros((faces.shape[0], self.num_classes))
        return self.infer_model.predict([normalized, dummy_y])