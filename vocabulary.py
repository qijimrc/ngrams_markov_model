#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os
import pickle
from collections import Counter, defaultdict
from typing import List
import copy
from tqdm import tqdm


class Vocabulary:

    def __init__(self, N: int, corpus: List[str]=None):
        # predefine special tokens
        self.unk = "<UNK>"
        self.bos = "<BOS>"
        self.eos = "<EOS>"
        self.gramsNumber = N
        # build vocabs
        if corpus is not None:
            self.uniVocab, self.ngramsFreqVocabs, self.ngramsProbsVocabs, \
                self.ngrams2Counts, self.count2Ngrams = self.build_ngrams_vocabs(N, corpus)


    def build_ngrams_vocabs(self, N: int, corpus: List[str]):
        """ Build an vocabulary with n-grams statistics

            Args
              - N: The number of n-grams.
              - corpus: A list of sentences with delimiter of balank space.
            Return
              - ngramsFreqVocabs: A dict for each key and value representing
                the grams number 'N' and it's counting matrix.
        """

        # statistics for all N-grams
        ngrams2Counts = defaultdict(Counter)
        for line in tqdm(corpus, desc="Build ngrams2Counts"):
            tokens = [self.bos] + line.strip().split() + [self.eos]
            for i in range(len(tokens)):
                # calculate
                j = i
                while i-j <= N-1 and j >= 0:
                    ngrams2Counts[i-j+1].update([tuple(tokens[j:i+1])])
                    j -= 1
        ## process `unk`
        #vocabLen = len(ngrams2Counts[1])
        #for n in range(1, N+1):
        #    if n==1: ngrams2Counts[n][tuple([self.unk])] = 0
        #    else:
        #        ngrams2Counts[n][tuple([self.unk]*n)] = vocabLen*vocabLen-len(ngrams2Counts[n])


        # calculate ngrams count with freq
        count2Ngrams = defaultdict(lambda:defaultdict(set))
        for n in ngrams2Counts:
            n2C = copy.copy(ngrams2Counts[n])
            sortedN2C = sorted(n2C.items(), key=lambda x:x[1])
            # build
            for ngram, count in tqdm(sortedN2C, desc="Build count2Ngrams for n="+str(n)):
                count2Ngrams[n][count].add(ngram)

        # build vocab matrix
        sortedUnigrams = sorted(ngrams2Counts[1].items(), key=lambda x:x[1], reverse=True)
        uniVocab = [self.unk] + [k[0] for k,v in sortedUnigrams]
        ngramsFreqVocabs = defaultdict(np.array)
        ngramsProbsVocabs = defaultdict(np.array)
        for n in ngrams2Counts:
            freqMatrix = np.zeros([len(uniVocab)]*n)
            # update
            for grams, count in tqdm(ngrams2Counts[n].items(), desc="Build vocabs for n="+str(n)):
                changeData = freqMatrix
                for k in range(len(grams)-1):
                    idx = uniVocab.index(grams[k])
                    changeData = changeData[idx]
                idx = uniVocab.index(grams[-1])
                changeData[idx] = count
            ngramsFreqVocabs[n] = freqMatrix
            ngramsProbsVocabs[n] = freqMatrix / freqMatrix.sum()

        return uniVocab, ngramsFreqVocabs, ngramsProbsVocabs, ngrams2Counts, count2Ngrams

    def save_model(self, saveDir):
        with open(os.path.join(saveDir, "uniVocab.pkl"), "wb") as f:
            pickle.dump(obj=self.uniVocab, file=f)
        with open(os.path.join(saveDir, "ngrams2Counts.pkl"), "wb") as f:
            pickle.dump(obj=self.ngrams2Counts, file=f)
        #with open(os.path.join(saveDir, "count2Ngrams.pkl"), "wb") as f:
        #    pickle.dump(obj=self.count2Ngrams, file=f)
        np.savez(os.path.join(saveDir, "ngramsFreqVocabs.npz"), *[v for k,v in self.ngramsFreqVocabs.items()])
        np.savez(os.path.join(saveDir, "ngramsProbsVocabs.npz"), *[v for k,v in self.ngramsProbsVocabs.items()])


    def load_model(self, saveDir):
        with open(os.path.join(saveDir, "uniVocab.pkl"), "wb") as f:
            self.uniVocab = pickle.load(f)
        with open(os.path.join(saveDir, "ngrams2Counts.pkl"), "wb") as f:
            self.ngrams2Counts=pickle.load(f)
        #with open(os.path.join(saveDir, "count2Ngrams.pkl"), "wb") as f:
        #    self.count2Ngrams=pickle.load(f)
        freqData = np.load(os.path.join(saveDir, "ngramsFreqVocabs.npz"))
        ProbsData = np.load(os.path.join(saveDir, "ngramsProbsVocabs.npz"))
        for n in self.ngramsFreqVocabs:
            self.ngramsFreqVocabs[n] = freqData["arr_"+str(n)]
        for n in self.ngramsProbsVocabs:
            self.ngramsProbsVocabs[n] = ProbsData["arr_"+str(n)]

    @classmethod
    def get_value(self, data: np.array, idxs: List[int]):
        if len(idxs) == 1:
            return data[idxs[0]]
        else:
            return self.get_value(data[idxs[0]], idxs[1:])

    @classmethod
    def get_indices(self, data: np.array, value: int, idxs: list, rt: list):
        import ipdb
        ipdb.set_trace()
        for i in range(len(data)):
            idxs.append(i)
            if len(data.shape) == 1:
                if data[i] == value:
                    rt.append(copy.copy(idxs))
                idxs.pop()
            else:
                get_indices(data[i], value, idxs, rt)

    def tune_with_Laplace_smoothing(self, B: int):
        """ Tune the all ngrams vocabularies with laplace smoothing (adding one) strategy.

            Args:
              - B: the parameter.
        """
        for n in self.ngramsFreqVocabs:
            totalN = sum(self.ngramsFreqVocabs[n].shape)
            self.ngramsProbsVocabs[n] = (self.ngramsFreqVocabs[n]+1)/(totalN+B)

        

    def tune_with_held_out_smoothing(self, heldOutData: List[str]):
        """ Tune the all ngrams vocabularies with held-out strategy.

            Args:
              - B: the parameter.
        """
        def assgin(freqArray, probArray, n2cInTrain, c2nInTrain, n2cInHeld):
            for i in tqdm(range(len(freqArray)),desc="Tuning with Held-Out Smoothing:"):
                if len(freqArray.shape) == 1:
                    r = freqArray[i]
                    if r ==0:
                        gramsSet = []
                        for ngh in n2cInHeld:
                            if ngh not in n2cInTrain.keys():
                                gramsSet.append(ngh)
                    else:
                        gramsSet = c2nInTrain[r]
                    N = len(n2cInHeld)
                    N_r = len(gramsSet)
                    T_r = sum([n2cInHeld[gram] for gram in gramsSet])
                    try:
                        probArray[i] = T_r/(N_r*N)
                    except:
                        import ipdb
                        ipdb.set_trace()
                else:
                    assgin(freqArray[i], probArray[i], n2cInTrain, c2nInTrain, n2cInHeld)

        # statistics for all N-grams on held-out data
        n2cInHeld = defaultdict(Counter)
        for line in heldOutData:
            tokens = [self.bos] + line.strip().split() + [self.eos]
            for i in range(len(tokens)):
                # calculate >1 grams
                j = i
                while i-j <= self.gramsNumber-1 and j >= 0:
                    n2cInHeld[i-j+1].update([tuple(tokens[j:i+1])])
                    j -= 1
        # tune
        assgin(self.ngramsFreqVocabs[self.gramsNumber],self.ngramsProbsVocabs[self.gramsNumber],self.ngrams2Counts[self.gramsNumber],
               self.count2Ngrams[self.gramsNumber], n2cInHeld[self.gramsNumber])
            

    def tune_with_cross_val_smoothing(self, heldOutData: List[str]):
        """ Tune the all ngrams vocabularies with held-out strategy.

            Args:
              - B: the parameter.
        """
        def assgin(data, c2nInTrain, n2cInTrain, c2nInHeld, n2cInHeld):
            for i in range(len(data)):
                if len(data.shape) == 1:
                    r = data[i]
                    gramsSet_1 = c2nInTrain[r]
                    gramsSet_2 = c2nInHeld[r]
                    N_1 = len(n2cInHeld)
                    N_r_1 = len(gramsSet_1)
                    T_r_1 = sum([n2cInHeld[gram] for gram in gramsSet_1])
                    N_2 = len(n2cInTrain)
                    N_r_2 = len(gramsSet_2)
                    T_r_2 = sum([n2cInTrain[gram] for gram in gramsSet_2])
                    data[i] = (T_r_1+T_r_2) / (N_1*N_r_1 + N_2+N_r_2)
                else:
                    assgin(data[i], c2nInTrain, n2cInTrain, c2nInHeld, n2cInHeld)

        # statistics for all N-grams on held-out data
        n2cInHeld = defaultdict(Counter)
        c2nInHeld = defaultdict(lambda: defaultdict(set))
        for line in heldOutData:
            tokens = [self.bos] + line.strip().split() + [self.eos]
            for i in range(len(tokens)):
                # calculate
                j = i
                while i-j >= self.gramsNumber-1 and j >= 0:
                    n2cInHeld[i-j+1].update([tuple(tokens[j:i+1])])
                    j -= 1
        for n in n2cInHeld:
            n2C = copy.copy(self.ngrams2Counts[n])
            sortedN2C = sorted(n2C.items(), lambda k,v:v)
            # build
            for ngram, count in sortedN2C:
                c2nInHeld[n][count].add(ngram)
        # tune
        assgin(self.ngramsProbsVocabs[self.gramsNumber],
               self.gramsNumber,
               self.count2Ngrams[self.gramsNumber],
               self.ngrams2Counts[self.gramsNumber],
               c2nInHeld[self.gramsNumber],
               n2cInHeld[self.gramsNumber])



    def tune_with_good_turing_smoothing(self, ):
        """ Tune the all ngrams vocabularies with held-out strategy.

            Args:
              - B: the parameter.
        """
        def assgin(data, c2nInTrain):
            for i in range(len(data)):
                if len(data.shape) == 1:
                    r = data[i]
                    N_0 = len(c2nInTrain[0])
                    N_1 = len(c2nInTrain[1])
                    N_r = len(c2nInTrain[r])
                    N_r1 = len(c2nInTrain[r+1])
                    if r > 0:
                        data[i] = (r+1)*(N_r1/N_r)
                    else:
                        data[i] = N_0/N_1
                else:
                    assgin(data[i], c2nInTrain)
        # tune
        assgin(self.ngramsProbsVocabs[self.gramsNumber], self.count2Ngrams[self.gramsNumber])



    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        return [self.uniVocab.index(tk) if tk in self.uniVocab 
                else self.uniVocab.index(self.unk) for tk in tokens]

    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        return [self.uniVocab[idx] for idx in ids]

