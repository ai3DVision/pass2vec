from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Embedding
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Flatten
from keras.layers.wrappers import TimeDistributedDense
from keras.optimizers import SGD
import numpy as np
import argparse
import math


def generatorModel(max_sequence_length, vocab_size, internal_representation_dim=500):
    model = Sequential()
    model.add(Dense(input_dim=100, output_dim=1024))
    model.add(RepeatVector(max_sequence_length))
    model.add(LSTM(internal_representation_dim, activation='sigmoid', inner_activation='hard_sigmoid', return_sequences=True))
    model.add(TimeDistributed(Dropout(0.5)))
    model.add(LSTM(internal_representation_dim, activation='sigmoid' inner_activation='hard_sigmoid', return_sequences=True))
    model.add(TimeDistributed(Dense(vocab_size)))
    model.add(TimeDistributed(Activation('softmax')))

    return model


def discriminatorModel(max_sequence_length, vocab_size, internal_representation_dim=500):
    model = Sequential()
    model.add(Embedding(vocab_size, internal_representation_dim, max_sequence_length))
    model.add(LSTM(internal_representation_dim, activation='sigmoid', inner_activation='hard_sigmoid', return_sequences=True))
    model.add(TimeDistributed(Dropout(0.5)))
    model.add(LSTM(internal_representation_dim, activation='sigmoid', inner_activation='hard_sigmoid', return_sequences=True))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model


def generatorContainingDiscriminator(generator, discriminator):
    model = Sequential()
    model.add(generator)
    discriminator.trainable = False
    model.add(discriminator)
    return model

def generateTrainingData(path_to_passwords):
    vocab = {}
    all_passwords = []
    max_len = 0
    with open(path_to_passwords, "r") as f:
        for line in f:
            password = line.rstrip()
            if len(password) > max_len:
                max_len = len(password)
            for char in password:
                if char in vocab.keys():
                    vocab[char] += 1
                else:
                    vocab[char] = 1
            all_passwords.append(password)
    #TODO Create character table and encode as 1-hot, then divide into train and test
    return X_train, X_test, (max_len, len(vocab))


def train(BATCH_SIZE, path_to_passwords):
    X_train, X_test, (max_sequence_length, vocab_size) = generateTrainingData(path_to_passwords)
    discriminator = discriminatorModel(max_sequence_length, vocab_size)
    generator = generatorModel(max_sequence_length, vocab_size)
    discriminator_on_generator = \
        generator_containing_discriminator(generator, discriminator)
    d_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    g_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    generator.compile(loss='binary_crossentropy', optimizer="SGD")
    discriminator_on_generator.compile(
        loss='binary_crossentropy', optimizer=g_optim)
    discriminator.trainable = True
    discriminator.compile(loss='binary_crossentropy', optimizer=d_optim)
    noise = np.zeros((BATCH_SIZE, 100))
    for epoch in range(100):
        print("Epoch is", epoch)
        print("Number of batches", int(X_train.shape[0]/BATCH_SIZE))
        for index in range(int(X_train.shape[0]/BATCH_SIZE)):
            for i in range(BATCH_SIZE):
                noise[i, :] = np.random.uniform(-1, 1, 100)
            password_batch = X_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
            generated_passwords = generator.predict(noise, verbose=0)
            #if index % 20 == 0: Maybe do something here to print predicted password
            X = np.concatenate((password_batch, generated_passwords))
            y = [1] * BATCH_SIZE + [0] * BATCH_SIZE
            d_loss = discriminator.train_on_batch(X, y)
            #print("batch %d d_loss : %f" % (index, d_loss))
            for i in range(BATCH_SIZE):
                noise[i, :] = np.random.uniform(-1, 1, 100)
            discriminator.trainable = False
            g_loss = discriminator_on_generator.train_on_batch(
                noise, [1] * BATCH_SIZE)
            discriminator.trainable = True
            #print("batch %d g_loss : %f" % (index, g_loss))
            if index % 10 == 9:
                generator.save_weights('generator', True)
                discriminator.save_weights('discriminator', True)


def generate(BATCH_SIZE, nice=False):
    generator = generator_model()
    generator.compile(loss='binary_crossentropy', optimizer="SGD")
    generator.load_weights('generator')
    if nice:
        discriminator = discriminator_model()
        discriminator.compile(loss='binary_crossentropy', optimizer="SGD")
        discriminator.load_weights('discriminator')
        noise = np.zeros((BATCH_SIZE*20, 100))
        for i in range(BATCH_SIZE*20):
            noise[i, :] = np.random.uniform(-1, 1, 100)
        generated_images = generator.predict(noise, verbose=1)
        d_pret = discriminator.predict(generated_images, verbose=1)
        index = np.arange(0, BATCH_SIZE*20)
        index.resize((BATCH_SIZE*20, 1))
        pre_with_index = list(np.append(d_pret, index, axis=1))
        pre_with_index.sort(key=lambda x: x[0], reverse=True)
        nice_images = np.zeros((BATCH_SIZE, 1) +
                               (generated_images.shape[2:]), dtype=np.float32)
        for i in range(int(BATCH_SIZE)):
            idx = int(pre_with_index[i][1])
            nice_images[i, 0, :, :] = generated_images[idx, 0, :, :]
        image = combine_images(nice_images)
    else:
        noise = np.zeros((BATCH_SIZE, 100))
        for i in range(BATCH_SIZE):
            noise[i, :] = np.random.uniform(-1, 1, 100)
        generated_images = generator.predict(noise, verbose=1)
        image = combine_images(generated_images)
    image = image*127.5+127.5
    Image.fromarray(image.astype(np.uint8)).save(
        "generated_image.png")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--nice", dest="nice", action="store_true")
    parser.add_argument("--path_to_passwords", type=str)
    parser.set_defaults(nice=False)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    if args.mode == "train":
        train(BATCH_SIZE=args.batch_size, path_to_passwords=args.path_to_passwords)
    elif args.mode == "generate":
        generate(BATCH_SIZE=args.batch_size, nice=args.nice)
