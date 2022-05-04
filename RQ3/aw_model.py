import tensorflow.keras as keras
import tensorflow as tf

from aw_models.aw_attendgru import AttentionGRUModel as attendgru
from aw_models.aw_astattendgru import AstAttentionGRUModel as ast_attendgru
from aw_models.aw_astattendgrufc import AstAttentiongruFCModel as ast_attendgru_fc 
from aw_models.aw_code2seq import Code2SeqModel as code2seq
from aw_models.aw_codegnngru import CodeGNNGRUModel as codegnngru
from aw_models.aw_transformer_base import TransformerBase as xformer_base

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
    else:
        print("{} is not a valid model type".format(modeltype))
        exit(1)
        
    return mdl.create_model()
