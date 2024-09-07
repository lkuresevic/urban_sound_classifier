import matplotlib.pyplot as plt
import csv

def plot_results(model_num):
    with open('model_' + str(model_num) + '_results.csv','r') as csvfile: 
        plots = csv.reader(csvfile, delimiter = ',') 
        
        epochs = [row[0] for row in data]
        loss = [row[1] for row in data]
        acc = [row[2] for row in data]
        test_loss = [row[3] for row in data]
        test_acc = [row[4] for row in data]
        
    plt.plot(epochs, loss, label='Training Loss', marker='o')
    plt.plot(epochs, test_loss, label='Test Loss', marker='o')
    
    plt.plot(epochs, acc, label='Training Accuracy', marker='o')
    plt.plot(epochs, test_acc, label='Test Accuracy', marker='o')

    plt.title('Training and Test Loss/Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    
    plt.legend()
    
    plt.show()