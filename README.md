# Destiny_RNN
Python Based RNN that creates guns.

# Requirements
* CUDA
* Theano/Tensorflow
* tqdm (for ProgressBars)
* nltk

# Instructions
1. You need generate the manifest. In the manifest directory, there is a python script (request.py) that will create the manifest.
2. Using the create.py script, you generate training data for the network
3. Setup Tensorflow/Theano
4. Run train.py/train_tensor.py

# Difference between Theano and Tensorflow
The Theano code was my first attempt to create the network. It was slow, and I had to do quite a bit of work to get the network to train in batches. Given that it was my first attempt, I am happy with the progress I was able to make on it.

I had to write some new functions and such for the tensorflow version. I have enjoyed working with that library.
