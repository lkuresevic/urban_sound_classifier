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
The UrbanSound8K dataset is a widely used benchmark for environmental
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
* • n fft: 1024 (size of fast Fourier transforms window)
* • hop length: 512 (distance between frames)
* • n mels: 128 (number of Mel bands)
* 
# 3. Model Selection
For this study, we selected both convolutional and recurrent neural network
architectures for their distinct strengths in capturing different aspects of audio
data.

**3.1 ResNet Architectures**

ResNet CNNs are known for their powerful feature extraction capabilities due to
their deep architectures and residual connections, and have proven to perform
well at image classification tasks. We considered four variants:
* • ResNet18: A relatively shallow network with 18 layers, often used as a
  baseline due to its lower computational complexity.
* • ResNet34: Aslightly deeper version, still employing basic residual blocks
  but with increased depth for better feature extraction.
  • ResNet50: Employs bottleneck blocks, increasing depth and feature rep
  resentation without a proportional increase in computation.
  • ResNet101: The deepest model in our comparison, with 101 layers and
  bottleneck blocks, aimed at extracting the most detailed features.
  
**3.1.1 Preliminary experiments**

During preliminary experiments, both ResNet50 and ResNet101 showed a ten
dency towards overfitting, with ResNet101 struggling to reach accuracy above
50% outside its training set. In the case of our relatively small dataset, this
opposed the hypothesis that more complex networks extract more detailed fea
tures. For this reason, we dropped ResNet101 out of consideration for final
training cycles, but stuck with ResNet50 in case that it generalizes better over
a longer training period.
Whencomparing efficiency of ResNet18 and ResNet34 over a smaller number
of training epochs, the simpler model converged faster once again, regardless of
different hyperparameters. Hence, ResNet18 was chosen to be trained in the
final experiment.

**3.2 Long Short-Term Memory (LSTMs) Networks**

LSTMs are designed to remember long-term dependencies, making them effec
tive at modeling sequential data, which audio is. For this task, we configured
the LSTM models with two different hidden layer sizes (64 and 128 units) and
used two LSTM layers:
  • LSTM (64 units, 2 layers): A smaller LSTM architecture to capture
  temporal dependencies with fewer parameters.
  • LSTM (128 units, 2 layers): A larger LSTM configuration to better
  capture complex dependencies in sound sequences.
  
**3.2.1 Preliminary experiments**

Having noticed a tendency of CNNs to overfit training data across trial experi
ments, we configured the LSTMs with a dropout rate of 0.2- this improved test
accuracy over a smaller number of epochs and effectively prevented the model
from overfitting training sets.
Introduction of more LSTM cells per model was considered, but ultimately
disregarded as it failed to show improvements that would justify the increase in
training time.

# Experimental Setup
placeholder
# Results
placeholder
# Discussion
placeholder
# Conclusion
placeholder
