import os
import sys
import traceback
import pickle
import h5py
import argparse
import collections
import random
import tensorflow as tf
import numpy as np

seed = 1337
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait, as_completed
import multiprocessing
from itertools import product
from multiprocessing import Pool
from timeit import default_timer as timer

def gendescr(model, batch, badfids, comseqpos, comstok, batchsize, config):
    
    comlen = config['comlen']
    
    fiddats = list(zip(*batch.values()))
    nfiddats = list()
    
    for fd in fiddats:
        fd = np.array(fd)
        nfiddats.append(fd)

    for i in range(1, comlen):
        results = model.predict(nfiddats, batch_size=batchsize)
        if(len(results)<=3):
            results = results[2]
        for c, s in enumerate(results):
            nfiddats[comseqpos][c][i] = np.argmax(s)

    final_data = {}
    for fid, com in zip(batch.keys(), nfiddats[comseqpos]):
        final_data[fid] = seq2sent(com, comstok)

    return final_data

def load_model_from_weights(modelpath, modeltype, datvocabsize, comvocabsize, smlvocabsize, datlen, comlen, smllen):
    config = dict()
    config['datvocabsize'] = datvocabsize
    config['comvocabsize'] = comvocabsize
    config['datlen'] = datlen # length of the data
    config['comlen'] = comlen # comlen sent us in workunits
    config['smlvocabsize'] = smlvocabsize
    config['smllen'] = smllen

    model = create_model(modeltype, config)
    model.load_weights(modelpath)
    return model

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('modelfile', type=str, default=None)
    parser.add_argument('--num-procs', dest='numprocs', type=int, default='4')
    parser.add_argument('--gpu', dest='gpu', type=str, default='')
    parser.add_argument('--data', dest='dataprep', type=str, default='/scratch/projects/opt-funcom/data/java90dataset/q90')
    parser.add_argument('--outdir', dest='outdir', type=str, default='outdir')
    parser.add_argument('--batch-size', dest='batchsize', type=int, default=200)
    parser.add_argument('--with-graph', dest='withgraph', action='store_true', default=False)
    parser.add_argument('--model-type', dest='modeltype', type=str, default=None)
    parser.add_argument('--outfile', dest='outfile', type=str, default=None)
    parser.add_argument('--dtype', dest='dtype', type=str, default='float32')
    parser.add_argument('--tf-loglevel', dest='tf_loglevel', type=str, default='3')
    parser.add_argument('--testval', dest='testval', type=str, default='test')
    parser.add_argument('--vmem-limit', dest='vmemlimit', type=int, default=0)

    args = parser.parse_args()
    
    outdir = args.outdir
    dataprep = args.dataprep
    modelfile = args.modelfile
    numprocs = args.numprocs
    gpu = args.gpu
    batchsize = args.batchsize
    modeltype = args.modeltype
    outfile = args.outfile
    testval = args.testval
    withgraph = args.withgraph
    vmemlimit = args.vmemlimit
    
    if outfile is None:
        outfile = modelfile.split('/')[-1]

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
    import tensorflow.keras.backend as K
    
    from model import create_model
    from myutils import prep, drop, batch_gen, seq2sent

    from custom.graphlayers import GCNLayer

    prep('loading sequences... ')
    extradata = pickle.load(open('%s/dataset_short.pkl' % (dataprep), 'rb'))
    h5data = h5py.File('%s/dataset_seqs.h5' % (dataprep), 'r')
    
    seqdata = dict()
    seqdata['dt%s' % testval] = h5data.get('/dt%s' % testval)
    seqdata['ds%s' % testval] = h5data.get('/ds%s' % testval)
    seqdata['s%s' % testval] = h5data.get('/s%s' % testval)
    seqdata['c%s' % testval] = h5data.get('/c%s' % testval)
    
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

    prep('loading config... ')
    (modeltype, mid, timestart) = modelfile.split('_')
    (timestart, ext) = timestart.split('.')
    modeltype = modeltype.split('/')[-1]
    config = pickle.load(open(outdir+'/histories/'+modeltype+'_conf_'+timestart+'.pkl', 'rb'))

    comlen = config['comlen']
    loc2fid = config['locfid']['c'+testval] # loc2fid[loc] = fid
    allfidlocs = list(loc2fid.keys())

    drop()

    prep('loading model... ')
    config, model = create_model(modeltype, config)
    model.load_weights(modelfile)
    #model = keras.models.load_model(modelfile, custom_objects={"tf":tf, "keras":keras, "GCNLayer":GCNLayer)
    print(model.summary())
    drop()

    comstart = np.zeros(comlen)
    stk = comstok.w2i['<s>']
    comstart[0] = stk
    outfn = outdir+"/predictions/predict-{}.txt".format(outfile.split('.')[0])
    outf = open(outfn, 'w')
    print("writing to file: " + outfn)
    batch_sets = [allfidlocs[i:i+batchsize] for i in range(0, len(allfidlocs), batchsize)]
 
    prep("computing predictions...\n")
    for c, fidloc_set in enumerate(batch_sets):
        st = timer()
        bg = batch_gen(h5data, extradata, testval, config, training=False)
        (batch, badfids, comseqpos) = bg.make_batch(fidloc_set)

        batch_results = gendescr(model, batch, badfids, comseqpos, comstok, batchsize, config)

        for key, val in batch_results.items():
            outf.write("{}\t{}\n".format(key, val))

        end = timer ()
        print("{} processed, {} per second this batch".format((c+1)*batchsize, batchsize/(end-st)))

    outf.close()        
    drop()
