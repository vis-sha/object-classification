import torch
from pathlib import Path

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

def eval(model , dataloader , loss_fn ,  metrics , device):

    loss , acc = 0,0
    for (x,y) in dataloader:
        x,y = x.to(device),y.to(device)
        y_pred = model(x)
        loss += loss_fn(y_pred , y)
        acc += metrics(y_true = y , y_pred = y_pred.argmax(dim=1))

    loss /= len(dataloader)
    acc /= len(dataloader)

    print(f"loss : {loss} , accuracy : {acc}")

def save_model(model,target_path,model_name):
    target_dir = Path(target_path)
    target_dir.mkdir(parents=True,exist_ok=True)
    saving_path = target_dir/model_name
    torch.save(obj = model.state_dict() , f = saving_path)
    print(f"Model save at : {saving_path}")
