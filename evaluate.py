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
    # find max
    diff = np.abs(argSortA - argSortB)
    maxDiffIdx = diff.argmax()
    diff[maxDiffIdx] = -1
    submaxDiffIdx = diff.argmax()

    maxDiffInfo = {
        "maxDiffIdx": maxDiffIdx,
        "submaxDiffIdx": submaxDiffIdx,
        "maxGramsProbsA" : allProbsA[maxDiffIdx],
        "maxGramsRankA" : argSortA[maxDiffIdx],
        "maxGramsProbsB" : allProbsB[maxDiffIdx],
        "maxGramsRankB" : argSortB[maxDiffIdx],
        "submaxGramsProbsA" : allProbsA[submaxDiffIdx],
        "submaxGramsRankA" : argSortA[submaxDiffIdx],
        "submaxGramsProbsB" : allProbsB[submaxDiffIdx],
        "submaxGramsRankB" : argSortB[submaxDiffIdx]
    }
    

    return rt, maxDiffInfo



