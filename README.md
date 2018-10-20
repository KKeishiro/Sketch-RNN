# Sketch-RNN

This repo contains the Keras code for `sketch-rnn`. You can refer to these excellent blog posts and the paper by David Ha if you are keen to gain deeper insights. I also provide a Jupyter notebook example for the demonstration.

* [Teaching Machines to Draw (updated version)](http://blog.otoro.net/2017/05/19/teaching-machines-to-draw/)

* [Teaching Machines to Draw (Google AI Blog)](https://ai.googleblog.com/2017/04/teaching-machines-to-draw.html)

* [A Neural Representation of Sketch Drawings (original paper)](https://arxiv.org/abs/1704.03477)

And also you can find the github repo in `Magenta project`.

* [Sketch-RNN: A Generative Model for Vector Drawings](https://github.com/tensorflow/magenta/tree/master/magenta/models/sketch_rnn)

![Model Schematic](https://github.com/KKeishiro/Sketch-RNN/blob/master/images/schematic_Sketch-RNN.svg)

# Installation

First, you will need to install `git`, if you do not have it yet.

Next, clone this repository by opening a terminal and typing the following commands:

```
$ cd $HOME # or any other development directory you prefer
$ git clone https://github.com/KKeishiro/Sketch-RNN.git
$ cd Sketch-RNN
```

If you do not want to install git, you can instead download this repo.

# Prerequisite

* Tensorflow 1.11.0

* Keras 2.2.2

* Python 3.6

# Training a Model

Even though you can find several datasets in `data` folder, I provide the pre-trained model weights only for owl dataset.

Here are some notes: The type of RNN cell is limited to LSTM, even though in the original implementation, you can also use LSTM cell with Layer Normalization and HyperLSTM. And also annealing the KL loss term is not implemented.

### Example Usage:

```
python train.py --data_dir=dataset_path --log_root=checkpoint_path [--resume_training --weights=weights_path]
```

For example,

```
python train.py --log_root=models/elephant
```
