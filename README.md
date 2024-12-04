# Introduction
Environmental sound classification (ESC) is a crucial component of computer
audition systems that aim to interpret everyday sounds. It finds applications in
areas such as smart homes, autonomous vehicles, and environmental monitoring,
where identifying sounds like sirens, dog barks, and car horns is essential.
In this study, we explore the UrbanSound8K dataset, which contains 8732
labeled audio samples from 10 different sound classes, including gun shots, chil
dren playing, and street music. We aim to compare the efficacy of deep learning
models, particularly convolutional and recurrent networks, in classifying these
environmental sounds. Specifically, we evaluate ResNet-based CNNs (ResNet18,
ResNet34, ResNet50, and ResNet101) and LSTMs to understand their relative
performance
# Dataset Overview
The [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html) dataset is a widely used benchmark for environmental
sound classification tasks. It contains 8732 audio files, each labeled as one
of 10 different classes. The dataset is organized into 10 folds, allowing for cross
validation. All excerpts are taken from field recordings uploaded to [www.freesound.org](https://freesound.org/),
hence the dataset is particularly challenging due to the high variability in sound
recordings, both in terms of acoustic environments and recording devices.
# Audio Preprocessing
For our experiments, each audio sample in the UrbanSound8K dataset was
converted into a Mel spectrogram. The spectrogram provides a time-frequency
representation of the sound, capturing both temporal dynamics and frequency
content, which are to be used for discriminating between different sound classes.
Key parameters include:
* n fft: 1024 (size of fast Fourier transforms window)
* hop length: 512 (distance between frames)
* n mels: 128 (number of Mel bands)

![Mel Spectogram Example](https://storage.googleapis.com/kagglesdsdata/datasets/4164536/7202367/archive/fold3/107228-5-0-3.png?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20241204%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20241204T225732Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=2512c8577fb010f4dff19fc9b7c9e1d57309453f5fa4eb96db6dce709e4e1d429187a7e4404056eac421091fb0b7cb717470f77cfdb4f92e1378f45c4d84c1edfa62811851616015e85aaae91b491c8c9105ed047dc8bcbd5003a95a8a12bc661a0330372e123c1753604f1a22a9cb9c21e09a31ce381c4f19beb17ea9b33fdd787e3d257d1276bf9dd8076c5d17dde99e9aa90fcf013485e9ec18e33436aceed4a8e52e26f8e85cf31388d3b1902a4e4d5ba4b4e4984497a844784c1f4ae371c8c79a0a40a2a7b15a45e6e66776259f81a2e05be233394e3e014b5af92ef0edcc0bcbf1a25d2bd7740c296c6426044679409008ef5b75e6d87a8a7282dc3aa4)
# Project Paper
You can read about the experiment in the [project paper](https://github.com/lkuresevic/urban_sound_classifier/blob/main/Comparing%20CNN%20and%20LSTM%20Architectures%20for%20Environmental%20Sound%20Classification.pdf)
