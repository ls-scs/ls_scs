import tensorflow.keras as keras
import tensorflow as tf

from models.attendgru import AttentionGRUModel as attendgru
from models.ast_attendgru import AstAttentionGRUModel as ast_attendgru
from models.ast_attendgru_fc import AstAttentionGRUFCModel as ast_attendgru_fc
from models.codegnngru import CodeGNNGRUModel as codegnngru
from models.code2seq import Code2SeqModel as code2seq
from models.transformer_base import TransformerBase as xformer_base
from models.qstransformer import QSTransformer as qs_xformer
from models.qstransformer2 import QSTransformer2 as qs_xformer2

def create_model(modeltype, config):
    mdl = None

    if modeltype == 'attendgru':
        mdl = attendgru(config)
    elif modeltype == 'ast-attendgru': 
        mdl = ast_attendgru(config)
    elif modeltype == 'ast-attendgru-fc':
        mdl = ast_attendgru_fc(config)
    elif modeltype == 'code2seq':
        mdl = code2seq(config)
    elif modeltype == 'codegnngru':
        mdl = codegnngru(config)
    elif modeltype == 'transformer-base':
        mdl = xformer_base(config)
    elif modeltype == 'transformer-ast':
        mdl = qs_xformer(config)
    elif modeltype == 'transformer-ast2':
        mdl = qs_xformer2(config)
    else:
        print("{} is not a valid model type".format(modeltype))
        exit(1)
        
    return mdl.create_model()
