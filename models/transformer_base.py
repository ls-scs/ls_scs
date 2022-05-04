import sys

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, Embedding, Reshape, GRU, Concatenate, LSTM, Dropout, BatchNormalization, Activation, concatenate, multiply, MaxPooling1D, Conv1D, Flatten, GlobalAveragePooling1D
from tensorflow.compat.v1.keras.layers import Bidirectional, RepeatVector, Permute, TimeDistributed, dot
from tensorflow.keras.optimizers import RMSprop, Adamax
from tensorflow.keras import utils, metrics

from custom.qstransformer_layers import EncoderBlock, DecoderBlock, TokenAndPositionEmbedding
from custom.ls_loss import custom_dist_cce_loss

class TransformerBase:
    def __init__(self, config):
        
        config['tdatlen'] = 50
        
        self.config = config
        self.tdatvocabsize = config['tdatvocabsize']
        self.comvocabsize = config['comvocabsize']
        self.datlen = config['tdatlen']
        self.comlen = config['comlen']
        self.lsfactor = config['lsfactor']
        
        self.embdims = 100
        self.attheads = 8 # number of attention heads
        self.recdims = 100 
        self.ffdims = 100 # hidden layer size in feed forward network inside transformer

        self.config['batch_config'] = [ ['tdat', 'com'], ['comout'] ]

    def create_model(self):
        
        dat_input = Input(shape=(self.datlen,))
        com_input = Input(shape=(self.comlen,))
        
        ee = TokenAndPositionEmbedding(self.datlen, self.tdatvocabsize, self.embdims)
        eeout = ee(dat_input)
        etransformer_block = EncoderBlock(self.embdims, self.attheads, self.ffdims)
        encout = etransformer_block(eeout, eeout)

        de = TokenAndPositionEmbedding(self.comlen, self.comvocabsize, self.embdims)
        deout = de(com_input)
        dtransformer_block = DecoderBlock(self.embdims, self.attheads, self.ffdims)
        decout = dtransformer_block(deout, encout)
        
        out = Flatten()(decout)
        out = Dense(self.comvocabsize, activation="softmax")(out)
        
        model = Model(inputs=[dat_input, com_input], outputs=out)
        lossf = custom_dist_cce_loss(self.lsfactor)

        model.compile(loss=lossf, optimizer='adam', metrics=['accuracy'], run_eagerly=True)
        return self.config, model
