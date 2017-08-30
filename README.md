# Convolutional Neural Network for Sentence Classification, in TensorFlow
This is a TensorFlow implementation of the model in Yoon Kim's [Convolutional Neural Networks for Sentence Classification](http://www.aclweb.org/anthology/D14-1181).

The model achieves high accuracy in classifying text into correct labels depending on the sentiment of the sentence.

The model uses Stanford Sentiment Treebank (SST-1) which has 5 different labels for training and evaluating.

Most of the code has been referenced.  The model was implemented to learn more about TensorFlow and supports Python 3.x and TensorFlow 1.x.

## Credits
Most part of the code were adapted from:
- [Denny Britz's git repository](https://github.com/dennybritz/cnn-text-classification-tf)
- [Amit Mandelbaum's git repository](https://github.com/mangate/ConvNetSent)

All credits go to them.

## Requirements
- Python 3.x
- NumPy
- TensorFlow 1.x
- Pandas
- six

## Usage

1) Clone the repository  via
```console
git clone https://github.com/sroyhong313/DeepMultiSentiment.git
```
2) Download Google's word embeddings binary file, extract it, and place it under `data/` folder

### Running:
```python
python train.py - -help
```
This lists all the flags and the variables that can be edited to train the model.

To train the model using default variables, run the following:

```python
python train.py
```

This will begin the training process. At every "checkpoint_every" value, the model will run on a cross-validation data.