#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import Counter, defaultdict
from typing import List, Tuple, Dict
import copy
from tqdm import tqdm
import numpy as np


class Vocabulary:

    def __init__(self, N: int, corpus: List[str]=None):
        # predefine special tokens
        self.unk = "<UNK>"
        self.bos = "<BOS>"
        self.eos = "<EOS>"
        self.gramsNumber = N
        # build vocabs
        if corpus is not None:
            self.uniVocab, self.ngrams2Counts, self.count2Ngrams, self.ngrams2Probs\
                = self.build_ngrams_vocabs(N, corpus)


    def build_ngrams_vocabs(self, N: int, corpus: List[str]):
        """ Build an vocabulary with n-grams statistics

            Args
              - N: The number of n-grams.
              - corpus: A list of sentences with delimiter of balank space.
            Return
        """
        # statistics for all N-grams
        uniVocab = Counter()
        ngrams2Counts = defaultdict(int)
        for line in tqdm(corpus, desc="Build ngrams2Counts"):
            tokens = [self.bos] + line.strip().split() + [self.eos]
            for i in range(len(tokens)):
                # calculate
                uniVocab.update(tuple([tokens[i]]))
                if i >= N-1:
                    ngrams2Counts[tuple(tokens[i-N+1:i+1])] += 1
        # process `unk`
        ngrams2Counts[tuple([self.unk]*N)] = 0


        # calculate ngrams count with freq
        count2Ngrams = defaultdict(set)
        for ngram in tqdm(ngrams2Counts, desc="Build count2Ngrams"):
            count2Ngrams[ngrams2Counts[ngram]].add(ngram)

        # build probs
        ngrams2Probs = defaultdict()
        totalCount = sum(ngrams2Counts.values())
        for ngrams in tqdm(ngrams2Counts, desc="Build ngrams2Probs"):
            prob = ngrams2Counts[ngrams] / float(totalCount)
            self.set_ngrams_prob(ngrams2Probs, ngrams, prob)

        sortedUnigrams = sorted(uniVocab.items(), key=lambda x:x[1], reverse=True)
        uniVocab = [self.unk] + [k for k,v in sortedUnigrams]
        # update probs for subgrams
        rt = self.update_all_subgrams_probs(ngrams2Probs, N)
        assert np.abs(rt -1) < 1e-6

        return uniVocab, ngrams2Counts, count2Ngrams, ngrams2Probs

    def set_ngrams_prob(self, gramsProbsDict: Dict, ngrams: Tuple, prob: float, cur: int=1):
        if cur == len(ngrams):
            if ngrams not in gramsProbsDict:
                gramsProbsDict[ngrams] = defaultdict()
            gramsProbsDict[ngrams]["curProb"] = prob
        else:
            if ngrams[:cur] not in gramsProbsDict:
                gramsProbsDict[ngrams[:cur]] = defaultdict()
                gramsProbsDict[ngrams[:cur]]["curProb"] = 0.
            self.set_ngrams_prob(gramsProbsDict[ngrams[:cur]], ngrams, prob, cur+1)


    def get_all_grams_prob_n(self, gramsProbsDict: Dict, N: int, rt: list):
        if N == 1:
            rt.append(gramsProbsDict["curProb"])
        else:
            for k in gramsProbsDict:
                if k != "curProb":
                    self.get_all_grams_prob_n(gramsProbsDict[k], N-1, rt)
        return rt

    def get_grams_prob_n(self, grams: Tuple[str]):

        def get_unk_prob(N):
            unkTok = tuple([self.unk]*N)
            tmpDict = self.ngrams2Probs
            for i in range(1, N+1):
                tmpDict = tmpDict[unkTok[:i]]
            return tmpDict["curProb"]

        tmpDict = self.ngrams2Probs
        for i in range(1, len(grams)+1):
            if grams[:i] not in tmpDict:
                # tmp = [x for x in grams]
                # tmp[i-1] = self.unk
                # grams = tuple(tmp)
                return get_unk_prob(len(grams))
            tmpDict = tmpDict[grams[:i]]
        return tmpDict["curProb"]

    def update_all_subgrams_probs(self, gramsProbsDict: Dict, N: int, cur: int=1):
        if cur <= N:
            totalProb = 0.
            for k in gramsProbsDict:
                if k != "curProb":
                    totalProb += self.update_all_subgrams_probs(gramsProbsDict[k], N, cur+1)
            gramsProbsDict["curProb"] = totalProb
            return totalProb
        else:
            return gramsProbsDict["curProb"]



    def tune_with_Laplace_smoothing(self, B: int):
        """ Tune the all ngrams vocabularies with laplace smoothing (adding one) strategy.

            Args:
              - B: the parameter.
        """
        totalN = sum(self.ngrams2Counts.values())
        for grams in tqdm(self.ngrams2Counts, "Tuning with Laplace smoothing"):
            prob = (self.ngrams2Counts[grams]+1) / (totalN+B)
            self.set_ngrams_prob(self.ngrams2Probs, grams, prob)
        self.update_all_subgrams_probs(self.ngrams2Probs, self.gramsNumber)


    def tune_with_held_out_smoothing(self, heldOutData: List[str]):
        """ Tune the all ngrams vocabularies with held-out strategy.

            Args:
              - B: the parameter.
        """

        # statistics for all N-grams on held-out data
        n2cInHeld = defaultdict(int)
        for line in tqdm(heldOutData, desc="Processing with Held-Out Smoothing"):
            tokens = [self.bos] + line.strip().split() + [self.eos]
            for i in range(len(tokens)):
                # calculate >1 grams
                if i >= self.gramsNumber-1:
                    n2cInHeld[tuple(tokens[i-self.gramsNumber+1:i+1])] += 1
        # tune
        vocabLen = len(self.uniVocab)
        totalNgramsCounts = sum(self.ngrams2Counts.values())
        N = sum(n2cInHeld.values())
        for r in tqdm(self.count2Ngrams, desc="Tuning with Held-Out Smoothing"):
            if r == 0:
                N_r = np.power(vocabLen, self.gramsNumber) - totalNgramsCounts
                T_r = 0
                for gramsH in n2cInHeld:
                    if gramsH not in self.ngrams2Counts:
                        T_r += n2cInHeld[gramsH]
            else:
                gramsSet = self.count2Ngrams[r]
                N_r = len(gramsSet)
                T_r = sum([n2cInHeld[g] for g in gramsSet])
            prob = T_r/(N_r*N)
            # assign
            gramsSet = self.count2Ngrams[r]
            for grams in gramsSet:
                self.set_ngrams_prob(self.ngrams2Probs, grams, prob)
        self.update_all_subgrams_probs(self.ngrams2Probs, self.gramsNumber)


            

    def tune_with_cross_val_smoothing(self, heldOutData: List[str]):
        """ Tune the all ngrams vocabularies with held-out strategy.

            Args:
              - B: the parameter.
        """
        # statistics for all N-grams on held-out data
        c2nInHeld = defaultdict(set)
        n2cInHeld = defaultdict(int)
        for line in tqdm(heldOutData, desc="Processing with Held-Out Smoothing"):
            tokens = [self.bos] + line.strip().split() + [self.eos]
            for i in range(len(tokens)):
                # calculate >1 grams
                if i >= self.gramsNumber-1:
                    n2cInHeld[tuple(tokens[i-self.gramsNumber+1:i+1])] += 1

        for grams in tqdm(n2cInHeld, desc="Processing with Held-Out Smoothing"):
            c2nInHeld[n2cInHeld[grams]].add(grams)
        # tune
        N_1 = sum(n2cInHeld.values())
        N_2 = sum(self.ngrams2Counts.values())
        for r in tqdm(self.count2Ngrams, desc="Tuning with cross-valid Smooth"):
            allGrams = self.count2Ngrams[r]
            if r == 0:
                T_r_1 = 0
                for gramsH in n2cInHeld:
                    if gramsH not in self.ngrams2Counts:
                        T_r_1 += n2cInHeld[gramsH]
                N_r_1 = np.power(len(self.ngrams2Counts), self.gramsNumber) - N_2
                T_r_2 = 0
                for gramsT in self.ngrams2Counts:
                    if gramsT not in n2cInHeld:
                        T_r_2 += self.ngrams2Counts[gramsT]
                N_r_2 = np.power(len(n2cInHeld), self.gramsNumber) - N_1
            else:
                gramsSet_1 = self.count2Ngrams[r]
                gramsSet_2 = c2nInHeld[r]
                T_r_1 = sum([n2cInHeld[g] for g in gramsSet_1])
                T_r_2 = sum([self.ngrams2Counts[g] for g in gramsSet_2])
                N_r_1 = len(self.count2Ngrams[r])
                N_r_2 = len(c2nInHeld[r])
            prob = (T_r_1+T_r_2) / (N_1*N_r_1 + N_2+N_r_2)
            # assign
            for grams in allGrams:
                self.set_ngrams_prob(self.ngrams2Probs, grams, prob)
        self.update_all_subgrams_probs(self.ngrams2Probs, self.gramsNumber)



    def tune_with_good_turing_smoothing(self, ):
        """ Tune the all ngrams vocabularies with held-out strategy.

            Args:
              - B: the parameter.
        """
        for r in tqdm(self.count2Ngrams, desc="Tuning with Good-Turing Smooth for"):
            N = sum(self.ngrams2Counts.values())
            N_0 = len(self.count2Ngrams[0])
            N_1 = len(self.count2Ngrams[1])
            if r in self.count2Ngrams:
                N_r = len(self.count2Ngrams[r])
            else: N_r = 0
            if r+1 in self.count2Ngrams:
                N_r1 = len(self.count2Ngrams[r+1])
            else: N_r1 = 0
            if r > 0:
                prob = (r+1)*(N_r1/N_r)/N
            else:
                prob = N_0/N_1/N
            gramsSet = self.count2Ngrams[r]
            for grams in gramsSet:
                self.set_ngrams_prob(self.ngrams2Probs, grams, prob)
        self.update_all_subgrams_probs(self.ngrams2Probs, self.gramsNumber)


