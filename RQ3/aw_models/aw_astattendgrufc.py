from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, Reshape, GRU, LSTM, Dropout, BatchNormalization, Activation, concatenate, multiply, MaxPooling1D, Conv1D, Flatten, Bidirectional, RepeatVector, Permute, TimeDistributed, dot
from tensorflow.keras.optimizers import RMSprop, Adamax
import tensorflow.keras as keras
import tensorflow.keras.utils
import tensorflow as tf
from tensorflow.keras import metrics

from custom.ls_loss import custom_dist_cce_loss
import tensorflow as tf

# Edited from the models generously made open source by Haque et. al.
# https://arxiv.org/abs/2004.04881

class AstAttentiongruFCModel:
    def __init__(self, config):
        config['sdatlen'] = 10
        config['stdatlen'] = 50
        
        config['tdatlen'] = 50

        config['smllen'] = 100
        config['3dsmls'] = False
        
        self.config = config
        self.tdatvocabsize = config['tdatvocabsize']
        self.comvocabsize = config['comvocabsize']
        self.smlvocabsize = config['smlvocabsize']
        self.tdatlen = config['tdatlen']
        self.sdatlen = config['sdatlen']
        self.comlen = config['comlen']
        self.smllen = config['smllen']
        self.lsfactor = config['lsfactor']

        self.config['batch_config'] = [ ['tdat', 'sdat', 'smlseq'], ['comout'] ]

        self.embdims = 100
        self.smldims = 100
        self.recdims = 100
        self.findims = 100

    def create_model(self):
        
        tdat_input = Input(shape=(self.tdatlen,))
        sdat_input = Input(shape=(self.sdatlen, self.config['stdatlen']))
        sml_input = Input(shape=(self.smllen,))
        
        tdel = Embedding(output_dim=self.embdims, input_dim=self.tdatvocabsize, mask_zero=False)
        tde = tdel(tdat_input)

        tenc = GRU(self.recdims, return_state=True, return_sequences=True)
        tencout, tstate_h = tenc(tde)

        se = Embedding(output_dim=self.smldims, input_dim=self.smlvocabsize, mask_zero=False)(sml_input)
        se_enc = GRU(self.recdims, return_state=True, return_sequences=True)
        seout, state_sml = se_enc(se)

        semb = TimeDistributed(tdel)
        sde = semb(sdat_input)

        senc = TimeDistributed(GRU(int(self.recdims)))
        senc = senc(sde)
        
        context = concatenate([senc, tencout, seout], axis=1)

        out = TimeDistributed(Dense(self.findims, activation="relu"))(context)

        out = Flatten()(out)
        out1 = Dense(self.comvocabsize, activation="softmax")(out)
        
        model = Model(inputs=[tdat_input, sdat_input, sml_input], outputs=out1)
        lossf = custom_dist_cce_loss(self.lsfactor)
        model.compile(loss=lossf, optimizer='adamax', metrics=['accuracy'], run_eagerly=True)
        model.optimizer.learning_rate=0.01
        return self.config, model

