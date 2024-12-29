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

![**Mel Spectogram Example**](https://github.com/lkuresevic/urban_sound_classifier/blob/main/mel_spectrogram_example.png)
# Repository Overview
* **.\Plots:** A folder containing plots of training and testing accuracies of each of the models trained during the experiment.
* **.\Results:** A folder containing .csv files storing experiment outcomes.
* **.\:** Python scripts.
