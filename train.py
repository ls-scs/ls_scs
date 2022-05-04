import pickle
import sys
import os
import math
import traceback
import argparse
import signal
import atexit
import time
import h5py

import random
import tensorflow as tf
import numpy as np

seed = 1337
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

from tensorflow.keras.callbacks import ModelCheckpoint, LambdaCallback, Callback
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
import tokenizer

class HistoryCallback(Callback):
    
    def setCatchExit(self, outdir, modeltype, timestart, mdlconfig):
        self.outdir = outdir
        self.modeltype = modeltype
        self.history = {}
        self.timestart = timestart
        self.mdlconfig = mdlconfig
        
        atexit.register(self.handle_exit)
        signal.signal(signal.SIGTERM, self.handle_exit)
        signal.signal(signal.SIGINT, self.handle_exit)
    
    def handle_exit(self, *args):
        if len(self.history.keys()) > 0:
            try:
                fn = outdir+'/histories/'+self.modeltype+'_hist_'+str(self.timestart)+'.pkl'
                histoutfd = open(fn, 'wb')
                pickle.dump(self.history, histoutfd)
                print('saved history to: ' + fn)
                
                fn = outdir+'/histories/'+self.modeltype+'_conf_'+str(self.timestart)+'.pkl'
                confoutfd = open(fn, 'wb')
                pickle.dump(self.mdlconfig, confoutfd)
                print('saved config to: ' + fn)
            except Exception as ex:
                print(ex)
                traceback.print_exc(file=sys.stdout)
        sys.exit()
    
    def on_train_begin(self, logs=None):
        self.epoch = []
        self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch.append(epoch)
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)


if __name__ == '__main__':

    timestart = int(round(time.time()))

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--gpu', type=str, help='0 or 1', default='0')
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=200)
    parser.add_argument('--epochs', dest='epochs', type=int, default=10)
    parser.add_argument('--model-type', dest='modeltype', type=str, default='vanilla')
    parser.add_argument('--with-graph', dest='withgraph', action='store_true', default=False)
    parser.add_argument('--action-words', dest='actionwords', action='store_true', default=False)
    parser.add_argument('--with-labelsmoothing', dest='withls', action='store_true', default=False)
    parser.add_argument('--ls-factor', dest='lsfactor', type=float, default=0)
    parser.add_argument('--vmem-limit', dest='vmemlimit', type=int, default=0)
    parser.add_argument('--data', dest='dataprep', type=str, default='/scratch/projects/opt-funcom/data/java90dataset/q90')
    parser.add_argument('--outdir', dest='outdir', type=str, default='outdir')
    parser.add_argument('--hops', dest='hops', type=int, default= 5)
    parser.add_argument('--dtype', dest='dtype', type=str, default='float32')
    parser.add_argument('--tf-loglevel', dest='tf_loglevel', type=str, default='3')
    parser.add_argument('--only-print-summary', dest='onlyprintsummary', action='store_true', default=False)
    args = parser.parse_args()
    
    outdir = args.outdir
    dataprep = args.dataprep
    gpu = args.gpu
    hops = args.hops
    batch_size = args.batch_size
    epochs = args.epochs
    modeltype = args.modeltype
    withgraph = args.withgraph
    withls = args.withls
    lsfactor = args.lsfactor
    vmemlimit = args.vmemlimit
    onlyprintsummary = args.onlyprintsummary

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = args.tf_loglevel
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                if(vmemlimit > 0):
                    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=vmemlimit)])
        except RuntimeError as e:
            print(e)

    import tensorflow.keras as keras
    import tensorflow.keras.utils
    import tensorflow.keras.backend as K

    from model import create_model

    K.set_floatx(args.dtype)
    from myutils import prep, drop, batch_gen, seq2sent

    prep('loading sequences... ')
    sqlfile = '{}/rawdats.sqlite'.format(dataprep)
    extradata = pickle.load(open('%s/dataset_short.pkl' % (dataprep), 'rb'))
    seqdata = h5py.File('%s/dataset_seqs.h5' % (dataprep), 'r')
    drop()

    if withgraph:
        prep('loading graph data... ')
        graphdata = pickle.load(open('%s/dataset_graph.pkl' % (dataprep), 'rb'))
        for k, v in extradata.items():
            graphdata[k] = v
        extradata = graphdata
        drop()

    prep('loading tokenizers... ')
    comstok = extradata['comstok']
    tdatstok = extradata['tdatstok']
    sdatstok = tdatstok
    smlstok = extradata['smlstok']
    if withgraph:
        graphtok = extradata['graphtok']
    drop()

    steps = int(np.array(seqdata.get('/ctrain').shape[0])/batch_size)
    valsteps = int(np.array(seqdata.get('/cval').shape[0])/batch_size)
    tdatvocabsize = tdatstok.vocab_size
    comvocabsize = comstok.vocab_size
    smlvocabsize = smlstok.vocab_size

    print('tdatvocabsize %s' % (tdatvocabsize))
    print('comvocabsize %s' % (comvocabsize))
    print('smlvocabsize %s' % (smlvocabsize))
    print('batch size {}'.format(batch_size))
    print('steps {}'.format(steps))
    print('training data size {}'.format(steps*batch_size))
    print('vaidation data size {}'.format(valsteps*100))
    print('------------------------------------------')

    config = dict()
    config['hops'] = hops
    config['lsfactor'] = lsfactor
    config['tdatvocabsize'] = tdatvocabsize
    config['comvocabsize'] = comvocabsize
    config['smlvocabsize'] = smlvocabsize

    try:
        config['fidloc'] = extradata['fidloc']
        config['locfid'] = extradata['locfid']
        config['comstok'] = extradata['comstok']
        config['comlen'] = int(np.array(seqdata.get('/ctrain')).shape[1])
        config['tdatlen'] = int(np.array(seqdata.get('/dttrain')).shape[1])
        config['sdatlen'] = extradata['config']['sdatlen']
        config['smllen'] = int(np.array(seqdata.get('/strain')).shape[1])
    except KeyError:
        pass # some configurations do not have all data, which is fine

    config['batch_size'] = batch_size
    prep('creating model... ')
    config, model = create_model(modeltype, config)
    drop()

    print(model.summary())
    if onlyprintsummary:
        sys.exit()

    gen = batch_gen(seqdata, extradata, 'train', config)
    checkpoint = ModelCheckpoint(outdir+'/models/'+modeltype+'_E{epoch:02d}_'+str(timestart)+'.h5')
    savehist = HistoryCallback()
    savehist.setCatchExit(outdir, modeltype, timestart, config)
    
    valgen = batch_gen(seqdata, extradata, 'val', config)
    callbacks = [ checkpoint, savehist ]

    try:
        history = model.fit(x=gen, steps_per_epoch=steps, epochs=epochs, verbose=1, max_queue_size=8, workers=1, use_multiprocessing=False, callbacks=callbacks, validation_data=valgen, validation_steps=valsteps)
    except Exception as ex:
        print(ex)
        traceback.print_exc(file=sys.stdout)