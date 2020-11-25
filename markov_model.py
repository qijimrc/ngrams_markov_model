#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from vocabulary import Vocabulary
from typing import List


class MarkovModel:

    def __init__(self, vocabulary: Vocabulary):
        self.vocab = vocabulary


    def predict_sent_prob(self,
            sent: str,
            gramsNumber: int) -> float:
        """ Calculate the probability of a sentence based on ngrams model.
        """ 
        tokens = [self.vocab.bos] + sent.strip().split() + [self.vocab.eos]
        rt_prob = 1.
        # process the first `gramsNumber-1` tokens
        for i in range(1, min(gramsNumber-1, len(tokens))):
            curTok = tuple([tokens[i]])
            grams = tuple(tokens[:i+1])
            rt_prob *= self.vocab.get_grams_probs(grams) /\
                self.vocab.get_grams_probs(curTok)
            rt_prob *= self.vocab.get_grams_candi_probs(tokens[i], grams)
        # process the rest
        for i in range(gramsNumber-1, len(tokens)):
            curTok = tuple([tokens[i]])
            grams = tuple(tokens[i-gramsNumber+1:i+1])
            rt_prob *= self.vocab.get_grams_probs(grams) /\
                self.vocab.get_grams_probs(curTok)
        rt_pp = np.power(rt_prob, -gramsNumber)
        return rt_prob, rt_pp


    def predict_topk_with_index(self,
            sent: str,
            gramsNumber: int,
            idx: int,
            topK: int) -> float:
        """ Calculate the probability of a sentence based on ngrams model.
        """ 
        assert idx >0
        tokens = [self.vocab.bos] + sent.strip().split() + [self.vocab.eos]
        rt = 1.
        # process the first `gramsNumber-1` tokens
        for i in range(1, min(gramsNumber-1, len(tokens))):
            grams = tuple(tokens[:i+1])
            curTok = tuple([tokens[i]])
            rt *= self.vocab.get_grams_probs(grams) /\
                self.vocab.get_grams_probs(curTok)
            # rt *= self.vocab.get_grams_candi_probs(tokens[i], grams)
            if i == idx:
                prevToks = grams[:i]
                allToksProbs = []
                for j in range(len(self.vocab.uniVocab)):
                    _curTok = self.vocab.uniVocab[j]
                    allToks = prevToks + tuple([_curTok])
                    gramProb = self.vocab.get_grams_probs(allToks)
                    allToksProbs.append((_curTok ,gramProb / self.vocab.get_grams_probs(curTok)))
                allToksProbs = sorted(allToksProbs, key=lambda x:x[1], reverse=True)
                rankCurTok = [x[0] for x in allToksProbs].index(tokens[i])
                allToksProbs = allToksProbs[:topK]
                return allToksProbs, (tokens[i], rt, rankCurTok)
        # process the rest
        for i in range(gramsNumber-1, len(tokens)):
            grams = tuple(tokens[i-gramsNumber+1:i+1])
            curTok = tuple([tokens[i]])
            rt *= self.vocab.get_grams_probs(grams) /\
                self.vocab.get_grams_probs(curTok)
            # rt *= self.vocab.get_grams_candi_probs(tokens[i], grams)
            if i == idx:
                prevToks = grams[:i]
                allToksProbs = []
                for j in range(len(self.vocab.uniVocab)):
                    _curTok = self.vocab.uniVocab[j]
                    allToks = prevToks + tuple([_curTok])
                    gramProb = self.vocab.get_grams_probs(allToks)
                    # candiProb = self.vocab.get_grams_candi_probs(tokens[i], allToks)
                    allToksProbs.append((_curTok ,gramProb / self.vocab.get_grams_probs(curTok)))
                allToksProbs = sorted(allToksProbs, key=lambda x:x[1], reverse=True)
                rankCurTok = [x[0] for x in allToksProbs].index(tokens[i])
                allToksProbs = allToksProbs[:topK]
                return allToksProbs, (tokens[i], rt, rankCurTok)



