import yaml
import torch
from utils.ModelGetter import ModelGetter
# load config
CONFIG_FILE = "config/config.yaml"

with open(CONFIG_FILE,"r",encoding="utf-8") as f:
    config = yaml.safe_load(f)

# load dataset


# load model
model_getter = ModelGetter(config)
model = model_getter.get_model("grid_regressor")
model = model.to(config['device'])
inputs = torch.rand(3,4)
score = model(inputs)
print(score)