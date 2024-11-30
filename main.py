import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from torch.utils.data import DataLoader
import csv
from torch.utils.data import Dataset
import torchaudio

from train import *
from us8k import US8K
from cnn import ResNet, Bottleneck, Block
from lstm import LSTM
from constants import *
from plot import plot_results

if __name__ == "__main__":
    #device set to CPU while data preprocessing is performed
    device = "cpu"
    
    #setting up data pre-processing
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=128
    )
    
    # instantiating dataset object
    usd = US8K(ANNOTATIONS_FILE,
                            AUDIO_DIR,
                            mel_spectrogram,
                            SAMPLE_RATE,
                            NUM_SAMPLES,
                            device)

    #device moved to GPU for training purposes
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    #criterion defined
    loss_fn = nn.CrossEntropyLoss()
    
    #menu
    print(f"Choose a model for training: ")
    print(f"(0) ResNet18")
    print(f"(1) ResNet34")
    print(f"(2) ResNet50")
    print(f"(3) LSTM (64 hidden, 2 units)")
    print(f"(4) LSTM (128 hidden, 2 units)")
    
    choice = int(input())
    models = []
    if choice == 0: #CNN - ResNet18
        model_0 = ResNet(Block, [2, 2, 2, 2], 10).to(device)
        optimizer = torch.optim.Adam(params=model_0.parameters(), lr=LEARNING_RATE)
        train(model_0, usd, loss_fn, optimizer, EPOCHS, device, "CNN_ResNet18")
    elif choice == 1: #CNN - ResNet34
        model_1 = ResNet(Block, [3, 4, 6, 3], 10).to(device)
        optimizer = torch.optim.Adam(params=model_1.parameters(), lr=LEARNING_RATE)
        train(model_1, usd, loss_fn, optimizer, EPOCHS, device, "CNN_ResNet34")
    elif choice == 2: #CNN - ResNet50
        model_2 = ResNet(Bottleneck, [3, 4, 6, 3], 10).to(device)
        optimizer = torch.optim.Adam(params=model_2.parameters(), lr=LEARNING_RATE)
        train(model_2, usd, loss_fn, optimizer, EPOCHS, device, "CNN_ResNet50")
    elif choice == 3: #LSTM (64 hidden, 2 layers)
        model_3 = LSTM(44, 64, 2, 10).to(device)
        optimizer = torch.optim.Adam(params=model_3.parameters(), lr=LEARNING_RATE)
        train(model_3, usd, loss_fn, optimizer, EPOCHS, device, "LSTM_64_units_2_layers")
    elif choice == 4: #LSTM (128 hidden, 2 layers)
        model_4 = LSTM(44, 128, 2, 10).to(device)
        optimizer = torch.optim.Adam(params=model_4.parameters(), lr=LEARNING_RATE)
        train(model_4, usd, loss_fn, optimizer, EPOCHS, device, "LSTM_128_units_2_layers")
    else:
        exit()
   
    print(f"Successfully trained")
    
   