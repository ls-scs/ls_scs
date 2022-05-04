import sys
import javalang
from timeit import default_timer as timer
import tensorflow.keras as keras
import tensorflow.keras.utils
import numpy as np
import tensorflow as tf
import networkx as nx
import random
import threading
import sqlite3

import tokenizer

start = 0
end = 0

def prep(msg):
    global start
    statusout(msg)
    start = timer()

def statusout(msg):
    sys.stdout.write(msg)
    sys.stdout.flush()

def drop():
    global start
    global end
    end = timer()
    sys.stdout.write('done, %s seconds.\n' % (round(end - start, 2)))
    sys.stdout.flush()

def index2word(tok):
	i2w = {}
	for word, index in tok.w2i.items():
		i2w[index] = word

	return i2w

def seq2sent(seq, tokenizer):
    sent = []
    check = index2word(tokenizer)
    for i in seq:
        sent.append(check[i])

    return(' '.join(sent))
            
class batch_gen(tensorflow.keras.utils.Sequence):
    def __init__(self, seqdata, extradata, tt, config, training=True):
        self.comvocabsize = config['comvocabsize']
        self.tt = tt
        self.batch_size = config['batch_size']

        self.extradata = extradata

        self.seqdata = dict()
        self.seqdata['dt%s' % tt] = seqdata.get('/dt%s' % tt)
        self.seqdata['ds%s' % tt] = seqdata.get('/ds%s' % tt)
        self.seqdata['s%s' % tt] = seqdata.get('/s%s' % tt)
        self.seqdata['c%s' % tt] = seqdata.get('/c%s' % tt)

        self.allfidlocs = list(range(0, np.array(self.seqdata['dt%s' % tt]).shape[0]))
        self.config = config
        self.aw = config['aw']['%sfw' % (tt)]
        self.training = training
        
        if not self.training:
            comstart = np.zeros(self.config['comlen'])
            stk = self.extradata['comstok'].w2i['<s>']
            comstart[0] = stk
            self.comstart = comstart
        
        random.shuffle(self.allfidlocs)

    def __getitem__(self, idx):
        start = (idx*self.batch_size)
        end = self.batch_size*(idx+1)
        batchfidlocs = self.allfidlocs[start:end]
        return self.make_batch(batchfidlocs)

    def make_batch(self, batchfidlocs):
        return self.divideseqs(batchfidlocs)

    def __len__(self):
        return int(np.ceil(np.array(self.seqdata['dt%s' % (self.tt)]).shape[0])/self.batch_size)

    def on_epoch_end(self):
        random.shuffle(self.allfidlocs)

    def divideseqs(self, batchfidlocs):
        tdatseqs = list()
        sdatseqs = list()
        smlseqs = list()
        smlpaths = list()
        smlnodes = list()
        smledges = list()
        callnodes = list()
        calledges = list()
        bios = list()
        comouts = list()
        badfids = list()

        fiddat = dict()
        
        with_tdats = False
        with_sdats = False
        with_smlseqs = False
        with_smlpaths = False
        with_smlnodes = False
        with_smledges = False

        comseqpos = 0
        
        for c, inp in enumerate(self.config['batch_config'][0]):
            if inp == 'tdat':
                with_tdats = True
            elif inp == 'sdat':
                with_sdats = True
            elif inp == 'smlseq':
                with_smlseqs = True
            elif inp == 'smlpath':
                with_smlpaths = True
            elif inp == 'smlnode':
                with_smlnodes = True
            elif inp == 'smledge':
                with_smledges = True

        for fidloc in batchfidlocs:

            fid = self.config['locfid']['c%s' % self.tt][fidloc]
            if with_tdats:
                wtdatseq = self.seqdata['dt%s' % self.tt][fidloc]
                wtdatseq = wtdatseq[:self.config['tdatlen']]

            if with_sdats:
                wsdatseq = self.seqdata['ds%s' % self.tt][fidloc]
                wsdatseq = wsdatseq[:self.config['sdatlen']]
                wsdatseq = np.delete(wsdatseq, slice(self.config['stdatlen'],None), 1)

            if with_smlseqs:
                wsmlseq = self.seqdata['s%s' % self.tt][fidloc]
                wsmlseq = wsmlseq[:self.config['smllen']]

            if with_smlnodes or with_smlpaths:
                wsmlnodes = self.extradata['s%s_nodes' % (self.tt)][fid]
                wsmlnodeslen = len(wsmlnodes)

                # crop/expand ast sequence
                wsmlnodes = wsmlnodes[:self.config['maxastnodes']]
                tmp = np.zeros(self.config['maxastnodes'], dtype='int32')
                tmp[:wsmlnodes.shape[0]] = wsmlnodes
                wsmlnodes = np.int32(tmp)

            if with_smledges or with_smlpaths:
                wsmledges = self.extradata['s%s_edges' % (self.tt)][fid]
                if (wsmledges.shape[0] > 1000):
                    badfids.append(fid)
                    continue
            
                # crop/expand ast adjacency matrix to dense
                wsmledges = np.asarray(wsmledges.todense())
                wsmledges = wsmledges[:self.config['maxastnodes'], :self.config['maxastnodes']]
                tmp = np.zeros((self.config['maxastnodes'], self.config['maxastnodes']), dtype='int32')
                tmp[:wsmledges.shape[0], :wsmledges.shape[1]] = wsmledges
                wsmledges = np.int32(tmp)

            if with_smlpaths:
                g = nx.from_numpy_matrix(wsmledges)
                astpaths = nx.all_pairs_shortest_path(g, cutoff=self.config['pathlen'])
                wsmlpaths = list()

                for astpath in astpaths:
                    source = astpath[0]
                    
                    if len([n for n in g.neighbors(source)]) > 1:
                        continue
                    
                    for path in astpath[1].values():
                        if len([n for n in g.neighbors(path[-1])]) > 1:
                            continue # ensure only terminals as in Alon et al
                        
                        if len(path) > 1 and len(path) <= self.config['pathlen']:
                            newpath = self.idx2tok(wsmlnodes, path)
                            tmp = [0] * (self.config['pathlen'] - len(newpath))
                            newpath.extend(tmp)
                            wsmlpaths.append(newpath)
                
                random.shuffle(wsmlpaths) # Alon et al stipulate random selection of paths
                wsmlpaths = wsmlpaths[:self.config['maxpaths']] # Alon et al use 200, crop/expand to size
                if len(wsmlpaths) < self.config['maxpaths']:
                    wsmlpaths.extend([[0]*self.config['pathlen']] * (self.config['maxpaths'] - len(wsmlpaths)))
                wsmlpaths = np.asarray(wsmlpaths)

            if not self.training:
                inps = list()
                for inp in self.config['batch_config'][0]:
                    if inp == 'tdat':
                        inps.append(wtdatseq)
                    elif inp == 'sdat':
                        inps.append(wsdatseq)
                    elif inp == 'smlseq':
                        inps.append(wsmlseq)
                    elif inp == 'smlpath':
                        inps.append(wsmlpaths)
                    elif inp == 'smlnode':
                        inps.append(wsmlnodes)
                    elif inp == 'smledge':
                        inps.append(wsmledges)
                fiddat[fid] = inps
            else:                
                if with_tdats:
                    tdatseqs.append(wtdatseq)
                if with_sdats:
                    sdatseqs.append(wsdatseq)
                if with_smlseqs:
                    smlseqs.append(wsmlseq)
                if with_smlpaths:
                    smlpaths.append(wsmlpaths)
                if with_smlnodes:
                    smlnodes.append(wsmlnodes)
                if with_smledges:
                    smledges.append(wsmledges)
                comout = keras.utils.to_categorical(self.aw[fid], num_classes=self.comvocabsize)
                comouts.append(np.asarray(comout))

        if self.training:
            if with_tdats:
                tdatseqs = np.asarray(tdatseqs)
            if with_sdats:
                sdatseqs = np.asarray(sdatseqs)
            if with_smlseqs:
                smlseqs = np.asarray(smlseqs)
            if with_smlpaths:
                smlpaths = np.asarray(smlpaths)
            if with_smlnodes:
                smlnodes = np.asarray(smlnodes)
            if with_smledges:
                smledges = np.asarray(smledges)
            comouts = np.asarray(comouts)
            
            inps = list()
            oups = list()
            
            for inp in self.config['batch_config'][0]:
                if inp == 'tdat':
                    inps.append(tdatseqs)
                elif inp == 'sdat':
                    inps.append(sdatseqs)
                elif inp == 'smlseq':
                    inps.append(smlseqs)
                elif inp == 'smlpath':
                    inps.append(smlpaths)
                elif inp == 'smlnode':
                    inps.append(smlnodes)
                elif inp == 'smledge':
                    inps.append(smledges)
        
            for oup in self.config['batch_config'][1]:
                if oup == 'comout':
                    oups.append(comouts)
                
            if len(oups) == 1:
                oups = oups[0]
        
        if not self.training:
            return (fiddat, badfids)
        else:
            return (inps, oups)

    def idx2tok(self, nodelist, path):
        out = list()
        for idx in path:
            out.append(nodelist[idx])
        return out
