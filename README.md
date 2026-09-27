# Weed Species Classification (DeepWeeds)

Image classification of 8 weed species (plus a negative class) from the [DeepWeeds](https://github.com/AlexOlsen/DeepWeeds) dataset, using transfer learning with EfficientNetB0. Built for CS 540: Artificial Intelligence.

## Approach

- **Data:** about 17.5k labelled field images of weeds in northern Australia, split 80/20 into train/validation.
- **Augmentation:** rescaling, rotation, zoom and shifts via `ImageDataGenerator`.
- **Model:** EfficientNetB0 pre-trained on ImageNet as a frozen feature extractor, with a small Conv2D/MaxPooling and dense classification head on top (9-way softmax).
- **Training:** SGD with categorical cross-entropy (224×224 inputs, batch size 16), and `ModelCheckpoint` keeping the best model by validation accuracy.

## Files

- `deepweeds.ipynb`: main notebook (EDA, class distribution, model, training)
- `index.ipynb`: alternate pipeline using `tensorflow_datasets`
- `train_set_labels.csv` / `test_set_labels.csv`: labels
- `images/`: dataset images

## Status

The training run saved in the notebook is partial: it was stopped early, at about 53% validation accuracy after the first epoch. Next steps are a full training run, unfreezing the top EfficientNet blocks for fine-tuning, and reporting per-class precision/recall.

## Tech

Python · TensorFlow / Keras · scikit-learn · pandas · seaborn
