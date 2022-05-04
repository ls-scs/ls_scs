import sys

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, Embedding, Reshape, Concatenate, Dropout, BatchNormalization, Activation, concatenate, multiply, MaxPooling1D, Conv1D, Flatten
from tensorflow.compat.v1.keras.layers import Bidirectional, RepeatVector, Permute, TimeDistributed, dot
from tensorflow.keras.optimizers import RMSprop, Adamax
from tensorflow.keras import utils, metrics

from custom.qstransformer_layers import EncoderBlock, DecoderBlock, TokenAndPositionEmbedding
from custom.ls_loss import custom_dist_cce_loss

class QSTransformer:
    def __init__(self, config):
        
        config['tdatlen'] = 50
        
        self.config = config
        self.tdatvocabsize = config['tdatvocabsize']
        self.comvocabsize = config['comvocabsize']
        self.smlvocabsize = config['smlvocabsize']
        self.datlen = config['tdatlen']
        self.comlen = config['comlen']
        self.smllen = config['smllen']
        self.lsfactor = config['lsfactor']
        
        self.embdims = 100
        self.attheads = 8 # number of attention heads
        self.recdims = 100 
        self.smldims = 100
        self.ffdims = 100 # hidden layer size in feed forward network inside transformer

        self.config['batch_config'] = [ ['tdat', 'com', 'smlseq'], ['comout'] ]

    def create_model(self):
        
        dat_input = Input(shape=(self.datlen,))
        com_input = Input(shape=(self.comlen,))
        sml_input = Input(shape=(self.smllen,))
        
        ee = TokenAndPositionEmbedding(self.datlen, self.tdatvocabsize, self.embdims)
        se = TokenAndPositionEmbedding(self.smllen, self.smlvocabsize, self.embdims)

        seout = se(sml_input)
        setransformer_block = EncoderBlock(self.embdims, self.attheads, self.ffdims)
        # sencout = setransformer_block(seout, seout)

        eeout = ee(dat_input)
        etransformer_block = EncoderBlock(self.embdims, self.attheads, self.ffdims)
        encout = etransformer_block(eeout, eeout)
        sencout = setransformer_block(seout, encout)

        de = TokenAndPositionEmbedding(self.comlen, self.comvocabsize, self.embdims)
        deout = de(com_input)
        dtransformer_block = DecoderBlock(self.embdims, self.attheads, self.ffdims)
        decout = dtransformer_block(deout, sencout)
        
        out = Flatten()(decout)
        out = Dense(self.comvocabsize, activation="softmax")(out)
        
        model = Model(inputs=[dat_input, com_input, sml_input], outputs=out)
        lossf = custom_dist_cce_loss(self.lsfactor)

        model.compile(loss=lossf, optimizer='adam', metrics=['accuracy'], run_eagerly=True)
        return self.config, model

