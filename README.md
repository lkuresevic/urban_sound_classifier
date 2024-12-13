# Introduction
Environmental sound classification (ESC) is a crucial component of computer
audition systems that aim to interpret everyday sounds. It finds applications in
areas such as smart homes, autonomous vehicles, and environmental monitoring,
where identifying sounds like sirens, dog barks, and car horns is essential.
In this study, we explore the **UrbanSound8K** dataset, which contains 8732
labeled audio samples from 10 different sound classes, including gun shots, chil
dren playing, and street music. We aim to compare the efficacy of deep learning
models, particularly convolutional and recurrent networks, in classifying these
environmental sounds. Specifically, we evaluate **ResNet**-based **CNNs** (ResNet18,
ResNet34, ResNet50, and ResNet101) and **LSTMs** to understand their relative
performance
# Project Paper
You can read about the experiment in the [project paper](https://github.com/lkuresevic/urban_sound_classifier/blob/main/Comparing%20CNN%20and%20LSTM%20Architectures%20for%20Environmental%20Sound%20Classification.pdf)
# Dataset Overview
The [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html) dataset is a widely used benchmark for environmental
sound classification tasks. It contains 8732 audio files, each labeled as one
of 10 different classes. The dataset is organized into 10 folds, allowing for cross
validation. All excerpts are taken from field recordings uploaded to [www.freesound.org](https://freesound.org/),
hence the dataset is particularly challenging due to the high variability in sound
recordings, both in terms of acoustic environments and recording devices.
# Audio Preprocessing
For our experiments, each audio sample in the **UrbanSound8K** dataset was
converted into a **Mel spectrogram**. The spectrogram provides a time-frequency
representation of the sound, capturing both temporal dynamics and frequency
content, which are to be used for discriminating between different sound classes.
Key parameters include:
* **n fft**: 1024 (size of fast Fourier transforms window)
* **hop length**: 512 (distance between frames)
* **n mels**: 128 (number of Mel bands)

![**Mel Spectogram Example**](https://storage.googleapis.com/kagglesdsdata/datasets/4164536/7202367/archive/fold1/102106-3-0-0.png?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20241213%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20241213T173453Z&X-Goog-Expires=345600&X-Goog-SignedHeaders=host&X-Goog-Signature=2fc6833345cfdcea0fcf2739ec79c9bc3e26eca678ec0db5e574718f9f4dfc34e9f3405bad8c1a6b3222fb4f127a4aa25d3aaabfa25974b4394961009ed9b76a9257b3d55ac3a81f2c2acb33f88d61fb8a2768cdf28b19b1f60ee7a5da65dee90fdee41c1d151c98505f5e1422f528fdf08c6bbf1b67b51739554f1ec7776b44c664500c2bb054876108ecbed9110650991e5c47fa8014cbc345feecc8112449af9c28d7640ecebac53729adf323782b4968c65822aa22e127ccffd5b4deefa134c517d3f885945a2aec25763dade0185c03d3c46e28ad2285208aa6dfb678f7a95e8c2065225c04ed5474db48340022d11635711d739a6a91ff1ee469d66f68)
# Repository Overview
* **.\Plots:** A folder containing plots of training and testing accuracies of each of the models trained during the experiment.
* **.\Results:** A folder containing .csv files storing experiment outcomes.
* **.\:** Python scripts.
