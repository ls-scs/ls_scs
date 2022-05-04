from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, Reshape, GRU, LSTM, Dropout, BatchNormalization, Activation, concatenate, multiply, MaxPooling1D, Conv1D, Flatten, Bidirectional, RepeatVector, Permute, TimeDistributed, dot
from tensorflow.keras.optimizers import RMSprop, Adamax
import tensorflow.keras as keras
import tensorflow.keras.utils
import tensorflow as tf
from tensorflow.keras import metrics

from custom.ls_loss import custom_dist_cce_loss
import tensorflow as tf

# Edited from the models generously made open source by LeClair et. al. and Haque et. al.
# https://arxiv.org/abs/1902.01954
# https://arxiv.org/abs/2004.04881

class AstAttentionGRUModel:
    def __init__(self, config):
        
        config['tdatlen'] = 50

        self.config = config
        self.tdatvocabsize = config['tdatvocabsize']
        self.comvocabsize = config['comvocabsize']
        self.smlvocabsize = config['smlvocabsize']
        self.tdatlen = config['tdatlen']
        self.comlen = config['comlen']
        self.smllen = config['smllen']
        self.lsfactor = config['lsfactor']
        
        self.embdims = 100
        self.smldims = 100
        self.recdims = 100
        self.findims = 100

        self.config['batch_config'] = [ ['tdat', 'smlseq'], ['comout'] ]

    def create_model(self):
        
        dat_input = Input(shape=(self.tdatlen,))
        sml_input = Input(shape=(self.smllen,))
        
        ee = Embedding(output_dim=self.embdims, input_dim=self.tdatvocabsize, mask_zero=False)(dat_input)
        se = Embedding(output_dim=self.smldims, input_dim=self.smlvocabsize, mask_zero=False)(sml_input)

        se_enc = GRU(self.recdims, return_state=True, return_sequences=True)
        seout, state_sml = se_enc(se)

        enc = GRU(self.recdims, return_state=True, return_sequences=True)
        encout, state_h = enc(ee, initial_state=state_sml)

        context = concatenate([seout, encout], axis=1)

        out = TimeDistributed(Dense(self.findims, activation="relu"))(context)

        out = Flatten()(out)
        out = Dense(self.comvocabsize, activation="softmax")(out)
        
        model = Model(inputs=[dat_input, sml_input], outputs=out)
        lossf = custom_dist_cce_loss(self.lsfactor)
        
        model.compile(loss=lossf, optimizer='adam', metrics=['accuracy'], run_eagerly=True)
        return self.config, model


