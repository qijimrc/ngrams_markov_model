#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import List, Tuple
import numpy as np




def spearman_rank_correlation_annalysis(
        allGrams: List[Tuple],
        vocabA,
        vocabB):
    """ Analysis the Spearman Rank Correlation
    """
    rt = 0.

    allProbsA = []
    allProbsB = []
    for grams in allGrams:
        # print(grams)
        # import ipdb
        # ipdb.set_trace()
        allProbsA.append(vocabA.get_grams_prob_n(grams))
        allProbsB.append(vocabB.get_grams_prob_n(grams))

    # sort
    argSortA = np.argsort(allProbsA)[::-1]
    argSortB = np.argsort(allProbsB)[::-1]
    # claculate
    rt = 1 - 6*((np.power((argSortA - argSortB), 2)).sum()) / (np.power(len(argSortA),3) - len(argSortA))
    return rt



