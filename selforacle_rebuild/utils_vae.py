import datetime
import os
import time

import numpy as np
import pandas as pd
import tensorflow
from sklearn.model_selection import train_test_split
from tensorflow import keras

from config import Config
from vae import Encoder, Decoder, VAE, Sampling
from utils_all import RESIZED_IMAGE_HEIGHT, RESIZED_IMAGE_WIDTH, IMAGE_CHANNELS, get_driving_styles
original_dim = RESIZED_IMAGE_HEIGHT * RESIZED_IMAGE_WIDTH * IMAGE_CHANNELS

def load_vae(cfg, load_vae_from_disk):
    """
    Load a trained VAE from disk and compile it, or creates a new one to be trained.
    """
    name = cfg.LOSS_SAO_MODEL + "-latent" + str(cfg.SAO_LATENT_DIM)

    if load_vae_from_disk:
        encoder = tensorflow.keras.models.load_model(cfg.SAO_MODELS_DIR + os.path.sep + 'encoder-' + name)
        decoder = tensorflow.keras.models.load_model(cfg.SAO_MODELS_DIR + os.path.sep + 'decoder-' + name)
        print("loaded trained VAE from disk")
    else:
        encoder = Encoder().call(cfg.SAO_LATENT_DIM, RESIZED_IMAGE_HEIGHT * RESIZED_IMAGE_WIDTH * IMAGE_CHANNELS)
        decoder = Decoder().call(cfg.SAO_LATENT_DIM, (cfg.SAO_LATENT_DIM,))
        print("created new VAE model to be trained")

    vae = VAE(model_name=name,
              loss=cfg.LOSS_SAO_MODEL,
              latent_dim=cfg.SAO_LATENT_DIM,
              encoder=encoder,
              decoder=decoder)
    vae.compile(optimizer=keras.optimizers.Adam(learning_rate=cfg.SAO_LEARNING_RATE))

    return vae, name

def load_data_for_vae_training(cfg, sampling=None):
    """
    Load training data_nominal and split it into training and validation set
    Load only the first lap for each track
    """
    start = time.time()

    x = None
    path = None
    x_train = None
    x_test = None

    data_df = pd.read_csv(cfg.FILEPATH)
    if sampling is not None:
        print("sampling every " + str(sampling) + "th frame")
        data_df = data_df[data_df.index % sampling == 0]

    if x is None:
        x = data_df[['path']].values
    else:
        x = np.concatenate((x, data_df[['path']].values), axis=0)

    try:
        x_train, x_test = train_test_split(x, test_size=cfg.TEST_SIZE, random_state=0)
    except TypeError:
        print("Missing header to csv files")
        exit()

    duration_train = time.time() - start
    print("Loading training set completed in %s." % str(datetime.timedelta(seconds=round(duration_train))))

    print("Data set: " + str(len(x)) + " elements")
    print("Training set: " + str(len(x_train)) + " elements")
    print("Test set: " + str(len(x_test)) + " elements")
    return x_train, x_test

# def load_vae_by_name(name):
#     """
#     Load a trained VAE from disk by name
#     """
#     cfg = Config()
#     cfg.from_pyfile("config_my.py")
#     print(cfg.SAO_MODELS_DIR + os.path.sep + 'encoder-' + name, cfg.SAO_MODELS_DIR + os.path.sep + 'decoder-' + name)
#     encoder = tensorflow.keras.models.load_model(cfg.SAO_MODELS_DIR + os.path.sep + 'encoder-' + name)
#     decoder = tensorflow.keras.models.load_model(cfg.SAO_MODELS_DIR + os.path.sep + 'decoder-' + name)

#     vae = VAE(model_name=name,
#               loss=cfg.LOSS_SAO_MODEL,
#               latent_dim=cfg.SAO_LATENT_DIM,
#               encoder=encoder,
#               decoder=decoder)
#     vae.compile(optimizer=keras.optimizers.Adam(learning_rate=cfg.SAO_LEARNING_RATE))

#     return vae
def load_vae_by_name(name):
    """
    Load a trained VAE from disk by name
    """
    try:
        cfg = Config()
        cfg.from_pyfile("config_my.py")
        path = "/bdata/usrdata/zjx/monitor_test/selforacle_rebuild/vae_ckpt"
        encoder_path = path + os.path.sep + 'encoder-' + name
        decoder_path = path + os.path.sep + 'decoder-' + name
        # encoder_path = cfg.SAO_MODELS_DIR + os.path.sep + 'encoder-' + name
        # decoder_path = cfg.SAO_MODELS_DIR + os.path.sep + 'decoder-' + name
        # encoder_path = "/home/zjx/ase22/sao/encoder-track1-MSE-latent2"
        # decoder_path = "/home/zjx/ase22/sao/decoder-track1-MSE-latent2"
        encoder = tensorflow.keras.models.load_model(encoder_path)
        # print(encoder.summary())
        decoder = tensorflow.keras.models.load_model(decoder_path)
        # print(decoder.summary())
        vae = VAE(model_name=name,
                loss=cfg.LOSS_SAO_MODEL,
                latent_dim=cfg.SAO_LATENT_DIM,
                encoder=encoder,
                decoder=decoder)
        vae.compile(optimizer=keras.optimizers.Adam(learning_rate=cfg.SAO_LEARNING_RATE))

        return vae
    except Exception as e:
        print(f"Error loading VAE model: {e}")
        return None