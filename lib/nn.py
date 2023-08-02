from tensorflow import keras
import os 

class NN():
        
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None

    def predict(self, X):
        if os.path.isfile(self.model_path):
            self.model = keras.models.load_model(self.model_path)
        else:
            raise Exception(f'Model not found in {self.model_path}')
        return self.model.predict(X)
    

