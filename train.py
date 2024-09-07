import torch
from torch import nn
from torch.utils.data import DataLoader

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100 
    return acc
    
def train(model, data_loader, loss_fn, optimizer, device, epochs, num):
    results = []

    for epoch in range(epochs):
        num = 0
        
        loss, acc, test_loss, test_acc = 0, 0, 0, 0
        for inpt, target in data_loader:
            if num % 20 == 0:
                print(f"{epoch}_{num}")
            inpt, target = inpt.to(device), target.to(device)
            if num < 1750:
                model.train()
                y_logits = model(inpt).squeeze()
                y_pred = torch.argmax(torch.round(y_logits), dim=1)
                
                loss_cur = loss_fn(y_logits, target) 
                loss += loss_cur
                acc += accuracy_fn(target,  y_pred) 
      
                optimizer.zero_grad()
                
                loss_cur.backward()
                
                optimizer.step()
                
            else:
                model.eval()
                with torch.inference_mode():
                    test_logits = model(inpt).squeeze()
                    test_pred = torch.argmax(torch.round(test_logits), dim=1)
                    
                    test_loss += loss_fn(test_logits, target) 
                    test_acc += accuracy_fn(target, test_pred) 
            num += 1
        
        loss = loss / 1750
        acc = acc / 1750
        test_loss = test_loss / 432
        test_acc = test_acc / 432
        
        
        print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")
            
    results_file = "Results\model_" + str(num) + "_results.csv"

    with open(results_file, "w", newline="") as file:
        csv_writer = csv.writer(file)
        csv_writer.writerows(results)