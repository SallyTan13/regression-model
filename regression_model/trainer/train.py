from ..utils.ModelGetter import ModelGetter
from ..data_process.MyDataset import MyDataset
from ..loss.MyLoss import MyLoss
from ..common import PROJECT_ROOT

import yaml
import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader


# load config
CONFIG_FILE = "regression_model/config/config.yaml"

with open(CONFIG_FILE,"r",encoding="utf-8") as f:
    config = yaml.safe_load(f)


writer = SummaryWriter("runs/" + config['model'] + "/")

# load dataset
print(PROJECT_ROOT.joinpath(config['train_data_path']))
train_dataset = MyDataset(PROJECT_ROOT.joinpath(config['train_data_path']))
train_dataloader = DataLoader(train_dataset,batch_size=config['batch_size'],shuffle=True,num_workers=3)
test_dataset = MyDataset(config['test_data_path'])
test_dataloader = DataLoader(test_dataset,batch_size=config['batch_size'],shuffle=False,num_workers=3)

# load model
model_getter = ModelGetter(config)
model = model_getter.get_model()
model = model.to(config['device'])

# load criterion and optimizer
criterion = MyLoss(config['loss_param']['k1'],config['loss_param']['k2'],config['loss_param']['k3'])
optim = torch.optim.SGD(model.parameters(),lr=1e-3)

# training process
for epoch in config['epoch']:
    print("-------epoch  {} -------".format(i+1))

    for i,[x,y] in enumerate(train_dataloader):
        y_ = model(x)
        loss = criterion(y_,y)
        optim.zero_grad()

        loss.backward()
        optim.step()

        train_step = len(train_dataloader)*epoch+i+1
        if train_step % 100 == 0:
            print("train time：{}, Loss: {}".format(train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), train_step)
        
    if (epoch + 1) % config['eval_epoch'] == 0:
        model.eval()
        total_test_loss = 0
        total_accuracy = 0
        with torch.no_grad():
            for x, y in test_dataloader: 
                outputs = model(x)
                loss = criterion(outputs, x)
                total_test_loss = total_test_loss + loss.item()
                accuracy = (outputs.argmax(1) == x).sum()
                total_accuracy = total_accuracy + accuracy
    
        print("test set Loss: {}".format(total_test_loss))
        print("test set accuracy: {}".format(total_accuracy/len(test_data)))
        writer.add_scalar("test_loss", total_test_loss, i)
        writer.add_scalar("test_accuracy", total_accuracy/len(test_data), i)
    if (epoch + 1) % config['save_epoch'] == 0:
        torch.save(model, "{}/model_{}.pth".format(config['model'],i+1))

    





