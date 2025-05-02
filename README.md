
# CycleGAN Horse2Zebra Translation 🐴➡️🦓

This project implements a Cycle-Consistent Adversarial Network (CycleGAN) to perform unpaired image-to-image translation between horses and zebras using TensorFlow. The goal is to convert images from the source domain (horses) to the target domain (zebras), and vice versa, without needing paired training data.

## Setup and Requirements

The project requires Python 3.7+, TensorFlow 2.x, TensorFlow Datasets, Matplotlib, and Seaborn.

## Dataset

We use the 'horse2zebra' dataset available from TensorFlow Datasets. It provides unpaired images of horses and zebras split into training and testing sets. The dataset is loaded with metadata and used to train two models for domain translation.

## Preprocessing

Images undergo several preprocessing steps:
- Random Jittering: Resize, crop, and horizontal flip to augment data.
- Normalization: Convert image pixel values to a range between -1 and 1 to stabilize GAN training.

## Model Architecture

The project uses two core models:
- Generator: A convolutional neural network that downsamples the input, passes it through a series of residual blocks, and upsamples to produce an output image in the target domain.
- Discriminator: A PatchGAN-style network that outputs a map of realism scores indicating whether each patch of the image is real or fake.

## Loss Functions

Training is guided by three types of losses:
- Adversarial Loss to ensure generated images look real to the discriminator.
- Cycle Consistency Loss to ensure translations are reversible (i.e., a horse turned into a zebra and back should still look like the original horse).
- Identity Loss to ensure images already in the target domain remain unchanged.

## Training Strategy

Two generators and two discriminators are trained simultaneously. Each step involves:
- Translating the image to the other domain and back.
- Calculating all losses.
- Updating generator and discriminator weights using optimizers.

## Visualization

The project visualizes input images, translated outputs, and discriminator responses during training. This helps track the model’s progress and verify that it is learning the correct domain mappings.

## Training

The model is trained over several epochs. At the end of each epoch, example translations are generated to visually assess performance. The number of training steps per epoch depends on the dataset size.

## Results

After training, the model can convincingly translate horse images to zebra images and vice versa, despite the data being unpaired. This demonstrates the power of cycle-consistent adversarial learning.

## References

- CycleGAN: Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks (Zhu et al., 2017)
- TensorFlow CycleGAN Tutorial
