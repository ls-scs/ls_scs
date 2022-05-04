from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, Reshape, GRU, LSTM, Dropout, BatchNormalization, Activation, concatenate, multiply, MaxPooling1D, Conv1D, Flatten, Bidirectional, RepeatVector, Permute, TimeDistributed, dot
from tensorflow.keras.optimizers import RMSprop, Adamax
import tensorflow.keras as keras
import tensorflow.keras.utils
import tensorflow as tf
from tensorflow.keras import metrics

from custom.ls_loss import custom_dist_cce_loss

# Edited from the models generously made open source by Haque. et. al.
# https://arxiv.org/abs/2004.04881

class Code2SeqModel:
    def __init__(self, config):

        config['sdatlen'] = 10
        config['stdatlen'] = 50

        config['tdatlen'] = 50

        config['smllen'] = 100
        config['3dsmls'] = False

        config['pathlen'] = 8
        config['maxpaths'] = 100
        
        self.config = config
        self.tdatvocabsize = config['tdatvocabsize']
        self.comvocabsize = config['comvocabsize']
        self.smlvocabsize = config['smlvocabsize']
        self.tdatlen = config['tdatlen']
        self.sdatlen = config['sdatlen']
        self.comlen = config['comlen']
        self.smllen = config['smllen']
        self.lsfactor = config['lsfactor']
        
        self.config['maxastnodes'] = config['maxpaths']

        self.config['batch_config'] = [ ['tdat', 'com', 'smlpath'], ['comout'] ]

        self.embdims = 100
        self.recdims = 100
        self.findims = 100


    def create_model(self):
        
        tdat_input = Input(shape=(self.tdatlen,))
        astp_input = Input(shape=(self.config['maxpaths'], self.config['pathlen']))
        
        tdel = Embedding(output_dim=self.embdims, input_dim=self.tdatvocabsize, mask_zero=False)
        tde = tdel(tdat_input)

        tenc = GRU(self.recdims, return_state=True, return_sequences=True)
        tencout, tstate_h = tenc(tde)

        aemb = TimeDistributed(tdel)
        ade = aemb(astp_input)
        
        aenc = TimeDistributed(GRU(int(self.recdims)))
        aenc = aenc(ade)
        
        context = concatenate([tencout, aenc], axis=1)
        out = TimeDistributed(Dense(self.findims, activation="relu"))(context)
        out = Flatten()(out)
        out1 = Dense(self.comvocabsize, activation="softmax")(out)
        
        model = Model(inputs=[tdat_input, astp_input], outputs=out1)
        
        lossf = custom_dist_cce_loss(self.lsfactor)

        model.compile(loss=lossf, optimizer='adamax', metrics=['accuracy'], run_eagerly=True)
        return self.config, model
