import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import PredefinedSplit
import pandas as pd
from math import ceil, floor
import csv

from constants import *

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100 
    return acc

def train_epoch(model, data_loader, loss_fn, optimizer, device):
    train_loss, train_acc = 0, 0
    model.train()
    it = 0
    for inpt, target in data_loader: 
        it +=1
        print(f"train_{it}")
        inpt, target = inpt.to(device), target.to(device)
        y_logits = model(inpt).squeeze()
        y_pred = torch.argmax(torch.round(y_logits), dim=1)
        
        loss_cur = loss_fn(y_logits, target)
        train_loss += loss_cur
        train_acc += accuracy_fn(target,  y_pred) 

        optimizer.zero_grad()
        
        loss_cur.backward()
        
        optimizer.step()
    return (train_loss, train_acc)
                
def test_epoch(model, data_loader, loss_fn, device):
    test_loss, test_acc = 0, 0
    model.eval()
    it = 0
    for inpt, target in data_loader:
        it +=1
        print(f"test_{it}")
        inpt, target = inpt.to(device), target.to(device)
        with torch.inference_mode():
            test_logits = model(inpt).squeeze()
            test_pred = torch.argmax(torch.round(test_logits), dim=1)
            
            test_loss += loss_fn(test_logits, target)
            test_acc += accuracy_fn(target, test_pred)
            
    return (test_loss, test_acc)

def train(model, dataset, loss_fn, optimizer, epochs, device, name):
    results = []
    
    metadata = pd.read_csv(ANNOTATIONS_FILE)
    folds = metadata['fold']-1
    kf = PredefinedSplit(test_fold = folds)
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):
        #instantiate new model for every fold
            
        train_loader = DataLoader(
            dataset=dataset,
            batch_size=BATCH_SIZE,
            sampler=torch.utils.data.SubsetRandomSampler(train_idx))
        test_loader = DataLoader(
            dataset=dataset,
            batch_size=BATCH_SIZE,
            sampler=torch.utils.data.SubsetRandomSampler(test_idx))    
 
        for epoch in range(epochs):
            train_loss, train_acc = train_epoch(model, train_loader, loss_fn, optimizer, device)
            test_loss, test_acc = test_epoch(model, test_loader, loss_fn, device)
                
            train_loss = train_loss / len(train_loader)
            train_acc = train_acc / len(train_loader)
            test_loss = test_loss / len(test_loader)
            test_acc = test_acc / len(test_loader)
            
            print(f"Fold: {fold+1} | Epoch: {epoch+1} | Loss: {train_loss:.5f}, Accuracy: {train_acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")
            results.append([fold+1, fold*epochs + epoch+1, train_loss.item(), train_acc, test_loss.item(), test_acc])
        
    results_file = "Results/" + name + "_results.csv"
    
    # eval_loader = DataLoader(
            # dataset=dataset,
            # batch_size=BATCH_SIZE)    
    # eval_loss, eval_acc = test_epoch(model, eval_loader, loss_fn, device)
    # eval_loss = eval_loss / len(eval_loader)
    # eval_acc = eval_acc / len(eval_loader)
    # results.append([eval_loss.item(), eval_acc])
        
    with open(results_file, "w", newline="") as file:
        csv_writer = csv.writer(file)
        csv_writer.writerows(results)   