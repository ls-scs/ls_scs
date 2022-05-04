from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Maximum, Dense, Embedding, Reshape, GRU, LSTM, Dropout, BatchNormalization, Activation, concatenate, multiply, MaxPooling1D, MaxPooling2D, Conv1D, Conv2D, Flatten, Bidirectional, RepeatVector, Permute, TimeDistributed, dot
from tensorflow.keras.optimizers import RMSprop, Adamax, Adam
import tensorflow.keras as keras
import tensorflow.keras.utils
import tensorflow.keras.backend as K
import tensorflow as tf

from aw_custom.graphlayers import GCNLayer
from aw_custom.ls_loss import custom_dist_cce_loss

# codegnngru baseline from ICPC'20 LeClair et al.

# sometimes called ast-attendgru-gnn

class CodeGNNGRUModel:
    def __init__(self, config):
        
        self.config = config

        self.config['tdatlen'] = 50

        self.tdatvocabsize = config['tdatvocabsize']
        self.comvocabsize = config['comvocabsize']
        self.astvocabsize = config['smlvocabsize']
        self.tdatlen = config['tdatlen']
        self.comlen = config['comlen']
        self.smllen = config['smllen']
        self.lsfactor = config['lsfactor']

        self.config['maxastnodes'] = config['smllen']
        
        self.config['batch_config'] = [ ['tdat', 'smlnode', 'smledge'], ['comout'] ]

        self.config['asthops'] = 2

        self.embdims = 100
        self.smldims = 100
        self.recdims = 100
        self.findims = 100

    def create_model(self):
        
        tdat_input = Input(shape=self.tdatlen)
        smlnode_input = Input(shape=self.smllen)
        smledge_input = Input(shape=(self.smllen, self.smllen))
        
        tdel = Embedding(output_dim=self.embdims, input_dim=self.tdatvocabsize, mask_zero=False)
        tde = tdel(tdat_input)
        se = tdel(smlnode_input)

        tenc = GRU(self.recdims, return_state=True, return_sequences=True)
        tencout, tstate_h = tenc(tde)

        astwork = se
        for k in range(self.config['asthops']):
            astwork = GCNLayer(self.embdims)([astwork, smledge_input])
        
        astwork = GRU(self.recdims, return_sequences=True)(astwork, initial_state=tstate_h)

        context = concatenate([tencout, astwork], axis=1)

        out = TimeDistributed(Dense(self.findims, activation="relu"))(context)

        out = Flatten()(out)
        out1 = Dense(self.comvocabsize, activation="softmax")(out)
        
        model = Model(inputs=[tdat_input, smlnode_input, smledge_input], outputs=out1)
        lossf = custom_dist_cce_loss(self.lsfactor)

        model.compile(loss=lossf, optimizer=Adam(lr=0.001, clipnorm=20.), metrics=['accuracy'], run_eagerly=True)
        return self.config, model
