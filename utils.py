import torch

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