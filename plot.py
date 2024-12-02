import matplotlib.pyplot as plt
import csv

def plot_results(model_name):
    epochs, loss, acc, test_loss, test_acc = [], [], [], [], []
    
    # Open and read the CSV file
    with open(f'Results/' + model_name + "_results.csv", 'r') as csvfile: 
        data = csv.reader(csvfile, delimiter=',')
        
        # Read each row and convert to the appropriate data type
        for row in data:
            try:
                epochs.append(int(row[1]))
                loss.append(float(row[2]))
                acc.append(float(row[3]))
                test_loss.append(float(row[4]))
                test_acc.append(float(row[5]))
            except ValueError:
                continue
    
    # Plotting the results
    plt.plot(epochs, loss, label='Training Loss', marker='o')
    plt.plot(epochs, test_loss, label='Test Loss', marker='o')
    plt.plot(epochs, acc, label='Training Accuracy', marker='o')
    plt.plot(epochs, test_acc, label='Test Accuracy', marker='o')

    plt.title('Training and Test Loss/Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'Plots/' + model_name +'_results.png')
    plt.close('all')