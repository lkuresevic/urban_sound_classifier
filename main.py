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
from cnn import CNN, ResNet, Bottleneck
from lstm import LSTM
from constants import *

def initialise_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    return train_dataloader


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
    
    train_dataloader = initialise_data_loader(usd, BATCH_SIZE)
    
    loss_fn = nn.CrossEntropyLoss()
    torch.manual_seed(42)
    
    models = []
    num = 1
    # train networks
    # 1. CNN - custom
    model_0 = CNN(input_size=1, hidden_size=16, output_size=10).to(device)
    optimizer = torch.optim.Adam(params=model_0.parameters(), lr=LEARNING_RATE)
    train(model_0, train_dataloader, loss_fn, optimizer, device, EPOCHS, num)
    num+=1
    # 2. CNN - ResNet50
    # model_1 = ResNet(Bottleneck, [3,4,6,3], 10, 1).to(device)
    # optimizer = torch.optim.Adam(params=model_1.parameters(), lr=LEARNING_RATE)
    # train(model_1, train_dataloader, loss_fn, optimizer, device, EPOCHS, num)
    # 3. CNN - ResNet101
    # model_2 = ResNet(Bottleneck, [3,4,23,3], num_classes, channels).to(device)
    # optimizer = torch.optim.Adam(params=model_0, lr=LEARNING_RATE)
    # 4. LSTM - 128 units
    # model_3 = LSTM(64, 256, 128, 10, device).to(device)
    # optimizer = torch.optim.Adam(params=model_3.parameters(), lr=LEARNING_RATE)
    # train(model_3, train_dataloader, loss_fn, optimizer, device, EPOCHS, num)
    # optimizer = torch.optim.Adam(params=model_0, lr=LEARNING_RATE)
    # 5. LSTM - 256 units
    # model_4 = LSTM(64, 256, 256, 10)
    # optimizer = torch.optim.Adam(params=model_0, lr=LEARNING_RATE)
    
    # train and eval model
    

    # save model
    # torch.save(model.state_dict(), "Models\model_0.pth")
    # models.append("Models\model_0.pth")
    # print("Trained CNN saved at: " + "Models\model_0.pth")
    #1. CNN - MojCNN
    
    #2. CNN - ResNet50
    
    #3. CNN - ResNet101
    
    #4. LSTM - 128 units, 1 Layer
    
    #5. LSTM - 256 units, 1 Layer
    
    
    
    #plot
    
    

    