

class Predictor:
    def __init__(self,model):
        self.model = model
        self.data = None

    def predict_recon(self,x):
