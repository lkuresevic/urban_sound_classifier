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

# def initialise_data_loader(train_data, batch_size):
    # train_dataloader = DataLoader(train_data, batch_size=batch_size)
    # return train_dataloader


if __name__ == "__main__":
    #device set to CPU while data preprocessing is performed
    device = "cpu"
    # instantiating dataset object and creating data loader
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    usd = US8K(ANNOTATIONS_FILE,
                            AUDIO_DIR,
                            mel_spectrogram,
                            SAMPLE_RATE,
                            NUM_SAMPLES,
                            device)
                            
    #device moved to GPU for purposes of training
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # train_dataloader = initialise_data_loader(usd, BATCH_SIZE)
    
    loss_fn = nn.CrossEntropyLoss()
    torch.manual_seed(42)
    
    print(f"Choose a model for training: ")
    print(f"(1) ResNet18")
    print(f"(2) ResNet34")
    print(f"(3) ResNet50")
    print(f"(4) ResNet101")
    print(f"(5) LSTM (128 hidden, 128 units)")
    print(f"(6) LSTM (256 hidden, 128 units)")
    
    choice = int(input())
    models = []
    if choice == 1: #CNN - ResNet18
        model_1 = ResNet(Block, [2, 2, 2, 2], 10).to(device)
        optimizer = torch.optim.Adam(params=model_1.parameters(), lr=LEARNING_RATE)
        train(model_1, usd, loss_fn, optimizer, EPOCHS, device)
    elif choice == 2: #CNN - ResNet34
        model_2 = ResNet(Bottleneck, [3, 4, 6, 3], 10).to(device)
        optimizer = torch.optim.Adam(params=model_2.parameters(), lr=LEARNING_RATE)
        train(model_2, usd, loss_fn, optimizer, EPOCHS, device)
    elif choice == 3: #CNN - ResNet50
        model_3 = ResNet(Bottleneck, [3, 4, 6, 3], 10).to(device)
        optimizer = torch.optim.Adam(params=model_3.parameters(), lr=LEARNING_RATE)
        train(model_3, usd, loss_fn, optimizer, EPOCHS, device)
    elif choice == 4: #CNN - ResNet101
        model_4 = ResNet(Bottleneck, [3, 4, 23, 3], 10).to(device)
        optimizer = torch.optim.Adam(params=model_4.parameters(), lr=LEARNING_RATE)
        train(model_4, usd, loss_fn, optimizer, EPOCHS, device)
    elif choice == 5: #LSTM (128 units, layer)
        model_5 = LSTM(44, 128, 128, 10).to(device)
        optimizer = torch.optim.Adam(params=model_5.parameters(), lr=LEARNING_RATE)
        train(model_5, usd, loss_fn, optimizer, EPOCHS, device)
    elif choice == 6: #LSTM (256 units, layer)
        model_6 = LSTM(44, 256, 128, 10).to(device)
        optimizer = torch.optim.Adam(params=model_6.parameters(), lr=LEARNING_RATE)
        train(model_6, usd, loss_fn, optimizer, EPOCHS, device)
    else:
        exit()

    print(f"Succesfully trained")