from model.GridRegressor import GridRegressor

class ModelGetter():
    def __init__(self,config):
        super(ModelGetter,self).__init__()
        self.config = config

    def get_model(self,model_name):
        if model_name == "grid_regressor":
            model = GridRegressor(self.config['input_dim'],self.config['hidden_dim'],self.config['num_class'])
        return model