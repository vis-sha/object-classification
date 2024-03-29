import torch
from tqdm import tqdm

def train_process(model,train_dataloader,loss_function,optimizer,metrics,device):

    train_loss , train_accuracy = 0,0
    model.train()
    for batch , (x,y) in enumerate(train_dataloader):
        x,y = x.to(device), y.to(device)

        y_pred_train = model(x)    # forward pass
        loss = loss_function(y_pred_train,y)    # calculate loss
        train_loss += loss    #update loss
        train_accuracy += metrics(y_true = y , y_pred = y_pred_train.argmax(dim=1))    # update accuracy
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss_per_batch = train_loss / len(train_dataloader)
    train_accuracy_per_batch = train_accuracy / len(train_dataloader)

    return train_loss_per_batch , train_accuracy_per_batch

def test_process(model,test_dataloader,loss_function,metrics,device):

    test_loss , test_accuracy = 0, 0
    model.eval()
    with torch.inference_mode():
        for (x,y) in test_dataloader:
            x,y = x.to(device), y.to(device)
            y_pred_test = model(x)
            loss = loss_function(y_pred_test,y)    # calculate loss
            test_loss += loss    #update loss
            test_accuracy += metrics(y_true = y , y_pred = y_pred_test.argmax(dim=1))    # update accuracy

    test_loss_per_batch = test_loss / len(test_dataloader)
    test_accuracy_per_batch = test_accuracy / len(test_dataloader)

    return test_loss_per_batch,test_accuracy_per_batch

def train(model,train_dataloader,test_dataloader,loss_function,optimizer,metrics,device,epochs):

    for epoch in tqdm(range(epochs)):

        train_loss_per_batch , train_accuracy_per_batch = train_process(
            model,
            train_dataloader,
            loss_function,
            optimizer,
            metrics,
            device)
        test_loss_per_batch,test_accuracy_per_batch = test_process(
            model,
            test_dataloader,
            loss_function,
            metrics,
            device)
        
        
        print(f"Epoch : {epoch+1}\n-------")
        print(f"Train loss: {train_loss_per_batch:.5f} | Train accuracy: {train_accuracy_per_batch:.2f}%\n")
        print(f"Test loss: {test_loss_per_batch:.5f} | Test accuracy: {test_accuracy_per_batch:.2f}%\n")




        



