# -*- coding: utf-8 -*-
'''An implementation of sequence to sequence learning for performing addition
Input: "535+61"
Output: "596"
Padding is handled by using a repeated sentinel character (space)
Input may optionally be inverted, shown to increase performance in many tasks in:
"Learning to Execute"
http://arxiv.org/abs/1410.4615
and
"Sequence to Sequence Learning with Neural Networks"
http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf
Theoretically it introduces shorter term dependencies between source and target.
Two digits inverted:
+ One layer LSTM (128 HN), 5k training examples = 99% train/test accuracy in 55 epochs
Three digits inverted:
+ One layer LSTM (128 HN), 50k training examples = 99% train/test accuracy in 100 epochs
Four digits inverted:
+ One layer LSTM (128 HN), 400k training examples = 99% train/test accuracy in 20 epochs
Five digits inverted:
+ One layer LSTM (128 HN), 550k training examples = 99% train/test accuracy in 30 epochs
'''

from __future__ import print_function
from keras.models import Sequential
from keras.engine.training import slice_X
from keras.layers import Activation, TimeDistributed, Dense, RepeatVector, recurrent
from keras.callbacks import EarlyStopping
import numpy as np
from six.moves import range


class CharacterTable(object):
    '''
    Given a set of characters:
    + Encode them to a one hot integer representation
    + Decode the one hot integer representation to their character output
    + Decode a vector of probabilities to their character output
    '''
    def __init__(self, chars, maxlen):
        self.chars = sorted(set(chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))
        self.maxlen = maxlen

    def encode(self, C, maxlen=None):
        maxlen = maxlen if maxlen else self.maxlen
        X = np.zeros((maxlen, len(self.chars) + 1))
        for i, c in enumerate(C):
            try:
                X[i, self.char_indices[c]] = 1
            except:
                #Unknown character encoding
                #TODO: Should be able to handle all characters
                X[i, -1] = 1
        return X

    def decode(self, X, calc_argmax=True):
        if calc_argmax:
            X = X.argmax(axis=-1)
        return ''.join(self.indices_char[x] for x in X)


class colors:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'

def longestPassword(path_to_pass_file):
    MAX = 0
    f = open(path_to_pass_file, "r")
    for line in f:
        if len(line.strip()) > MAX:
            MAX = len(line.strip()) 
    f.close()
    return MAX



# Parameters for the model and dataset
TRAINING_SIZE = 50000
DIGITS = 3
INVERT = True
path_to_pass_file = "/home/user/misc_github_projects/pass2vec/data/SecLists/Passwords/10_million_password_list_top_1000000.txt"
# Try replacing GRU, or SimpleRNN
RNN = recurrent.LSTM
HIDDEN_SIZE = 128
BATCH_SIZE = 128
LAYERS = 1
#MAXLEN = DIGITS + 1 + DIGITS
MAXLEN = longestPassword(path_to_pass_file)

chars = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ~`!@#$%^&*()_-+={[}]"\'/?:;<,>.\\  '
ctable = CharacterTable(chars, MAXLEN)


def ingestData(path_to_pass_file):
    input_seq = []
    target_seq = []
    with open(path_to_pass_file) as f:
        for line in f.readlines():
            seq = line.strip() + ' ' * (MAXLEN - len(line.strip()))
            if INVERT:
                input_seq.append(seq[::-1])
            else:
                input_seq.append(seq)
            target_seq.append(seq)
    return input_seq, target_seq

def vectorizeData(input_seq, target_seq):
    X = np.zeros((len(input_seq), MAXLEN, len(chars)), dtype=np.bool)
    y = np.zeros((len(target_seq), MAXLEN, len(chars)), dtype=np.bool)
    for i, password in enumerate(input_seq):
        X[i] = ctable.encode(password, maxlen=MAXLEN)
    for i, password in enumerate(target_seq):
        y[i] = ctable.encode(password, maxlen=MAXLEN)
    indices = np.arange(len(y))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    # Explicitly set apart 10% for validation data that we never train over
    split_at = len(X) - len(X) / 10
    (X_train, X_val) = (slice_X(X, 0, split_at), slice_X(X, split_at))
    (y_train, y_val) = (y[:split_at], y[split_at:])
    return X_train, y_train, X_val, y_val

def generateModel():
    model = Sequential()
    # "Encode" the input sequence using an RNN, producing an output of HIDDEN_SIZE
    # note: in a situation where your input sequences have a variable length,
    # use input_shape=(None, nb_feature).
    model.add(RNN(HIDDEN_SIZE, input_shape=(MAXLEN, len(chars))))
    # For the decoder's input, we repeat the encoded input for each time step
    model.add(RepeatVector(MAXLEN))
    # The decoder RNN could be multiple layers stacked or a single layer
    for _ in range(LAYERS):
        model.add(RNN(HIDDEN_SIZE, return_sequences=True))

    # For each of step of the output sequence, decide which character should be chosen
    model.add(TimeDistributed(Dense(len(chars))))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

def trainModel(model, X_train, y_train, X_val, y_val):
    # Train the model each generation and show predictions against the validation dataset
    early_stopping = EarlyStopping(monitor='val_loss', patience=2)
    model.fit(X_train, y_train, batch_size=BATCH_SIZE, nb_epoch=50,
              validation_data=(X_val, y_val), callbacks=[early_stopping])
    """
    for iteration in range(1, 200):
        print()
        print('-' * 50)
        print('Iteration', iteration)
        model.fit(X_train, y_train, batch_size=BATCH_SIZE, nb_epoch=1,
                  validation_data=(X_val, y_val))
        ###
        # Select 10 samples from the validation set at random so we can visualize errors
        for i in range(10):
            ind = np.random.randint(0, len(X_val))
            rowX, rowy = X_val[np.array([ind])], y_val[np.array([ind])]
            preds = model.predict_classes(rowX, verbose=0)
            q = ctable.decode(rowX[0])
            correct = ctable.decode(rowy[0])
            guess = ctable.decode(preds[0], calc_argmax=False)
            print('Q', q[::-1] if INVERT else q)
            print('T', correct)
            print(colors.ok + '☑' + colors.close if correct == guess else colors.fail + '☒' + colors.close, guess)
            print('---')"""
    return model

def saveModel(model):
    model.save_weights("model_weights.h5")
    model_json = model.to_json()
    f = open("model_architecture.json", "w")
    f.write(model_json)
    

def main():
    input_seq, target_seq = ingestData(path_to_pass_file)
    print("Number of training passwords: {0}".format(len(input_seq)))
    X_train, y_train, X_val, y_val = vectorizeData(input_seq, target_seq)
    print("Training data shapes: {0}\t{1}\t{2}\t{3}".format(X_train.shape, y_train.shape, X_val.shape, y_val.shape))
    model = generateModel()
    model = trainModel(model, X_train, y_train, X_val, y_val)
    saveModel(model)

if __name__ == "__main__":
    main()
