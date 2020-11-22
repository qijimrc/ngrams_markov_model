#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from vocabulary import Vocabulary
from markov_model import MarkovModel
from evaluate import spearman_rank_correlation_annalysis
import copy
import pickle
import argparse
import os

def train(trainFile, devFile, gramsNumber,
        smoothStrategy, BLaplace, vocabDir):
    # process data
    with open(trainFile, "r") as f:
        corpusTrain = f.readlines()
    with open(devFile, "r") as f:
        corpusDev = f.readlines()
    with open(devFile, "r") as f:
        corpusTrainDev = corpusTrain + corpusDev

    vocab = Vocabulary(gramsNumber, corpusTrainDev)
    vocabSmoothLaplace = copy.copy(vocab)
    vocabSmoothHeldOut = copy.copy(vocab)
    vocabSmoothCrossValid = copy.copy(vocab)
    vocabSmoothGoodTuring = copy.copy(vocab)

    vocabSmoothLaplace = vocabSmoothLaplace.tune_with_Laplace_smoothing(BLaplace)
    with open(os.path.join(vocabDir, "vocabSmoothLaplace.pkl"), "w") as f:
        pickle.dump(obj=vocabSmoothLaplace, file=f)

    vocabSmoothHeldOut = vocabSmoothHeldOut.tune_with_held_out_smoothing(corpusDev)
    with open(os.path.join(vocabDir, "vocabSmoothHeldOut.pkl"), "w") as f:
        pickle.dump(obj=vocabSmoothHeldOut, file=f)

    vocabSmoothCrossValid = vocabSmoothCrossValid.tune_with_cross_val_smoothing(corpusDev)
    with open(os.path.join(vocabDir, "vocabSmoothCrossValid.pkl"), "w") as f:
        pickle.dump(obj=vocabSmoothCrossValid, file=f)

    vocabSmoothGoodTuring = vocabSmoothGoodTuring.tune_with_good_turing_smoothing()
    with open(os.path.join(vocabDir, "vocabSmoothGoodTuring.pkl"), "w") as f:
        pickle.dump(obj=vocabSmoothGoodTuring, file=f)

    


def predidct_file(model, gramsGumber, testFile, saveFile):
    # predict
    with open(testFile, "r") as f:
        testData = f.readlines()
    sentProb, sentPP = model.predict_sent_prob(testData, gramsGumber)
    print(sentProb, sentPP)
    # save results
    with open(saveFile, "w") as f:
        f.write(sentPP)

def predict_sent(model, gramsNumber, sent, idx, topK, saveFile):
    rt = model.predict_topk_with_index(sent, gramsNumber, idx, topK)
    print(rt)
    # save results
    with open(saveFile, "w") as f:
        f.write(rt)

def analysis_with_spearman_rank_correlation(uniVocab, ngramsProbsVocabsA, ngramsProbsVocabsB, gramsNumber):
    rt = spearman_rank_correlation_annalysis(uniVocab, ngramsProbsVocabsA, ngramsProbsVocabsB, gramsNumber)
    return rt





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--func", type=str, default="train")
    parser.add_argument("--train_file", type=str, default="../corpus/s3/train.txt")
    parser.add_argument("--dev_file", type=str, default="../corpus/s3/valid.txt")
    parser.add_argument("--test_file", type=str, default="../corpus/s3/test.txt")
    parser.add_argument("--sent", type=str, default="原定 计划 是 早晨 在 一 座 山 上 吃 午饭")
    parser.add_argument("--idx", type=int, default=1)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--vocab_dir", type=str, default="")
    parser.add_argument("--out_dir", type=str, default="")
    parser.add_argument("--grams_number", type=int, default=2)
    parser.add_argument("--smooth_strategy", type=str, default="Laplace")
    parser.add_argument("--B_laplace", type=int, default=1)
    args = parser.parse_args()

    if args.func == "train":
        train(args.train_file,
            args.dev_file,
            args.grams_number,
            args.smooth_strategy,
            args.B_laplace,
            args.vocab_dir)
    else:
        # load models
        with open(os.path.join(args.vocabDir, "vocabSmoothLaplace.pkl"), "w") as f:
            vocabSmoothLaplace = pickle.load(file=f)
            modelLaplace = MarkovModel(vocabSmoothLaplace)
        with open(os.path.join(args.vocabDir, "vocabSmoothHeldOut.pkl"), "w") as f:
            vocabSmoothHeldOut = pickle.load(file=f)
            modelHeldOut = MarkovModel(vocabSmoothHeldOut)
        with open(os.path.join(args.vocabDir, "vocabSmoothCrossValid.pkl"), "w") as f:
            vocabSmoothCrossValid = pickle.load(file=f)
            modelCrossValid = MarkovModel(vocabSmoothCrossValid)
        with open(os.path.join(args.vocabDir, "vocabSmoothGoodTuring.pkl"), "w") as f:
            vocabSmoothGoodTuring = pickle.load(file=f)
            modelGoddTuring = MarkovModel(vocabSmoothGoodTuring)
        
        if args.func == "pred_file":
            predidct_file(modelLaplace, args.gramsGumber, args.testFile, os.path.join(args.out_dir, "modelLaplace.txt"))
            predidct_file(modelHeldOut, args.gramsGumber, args.testFile, os.path.join(args.out_dir, "modelHeldOut.txt"))
            predidct_file(modelCrossValid, args.gramsGumber, args.testFile, os.path.join(args.out_dir, "modelCrossValid.txt"))
            predidct_file(modelGoddTuring, args.gramsGumber, args.testFile, os.path.join(args.out_dir, "modelGoddTuring.txt"))
        elif args.func == "pred_sent":
            predict_sent(modelLaplace, args.gramsGumber, args.sent, args.idx, args.topK, os.path.join(args.out_dir, "modelLaplace.txt"))
            predict_sent(modelHeldOut, args.gramsGumber, args.sent, args.idx, args.topK, os.path.join(args.out_dir, "modelHeldOut.txt"))
            predict_sent(modelCrossValid, args.gramsGumber, args.sent, args.idx, args.topK, os.path.join(args.out_dir, "modelCrossValid.txt"))
            predict_sent(modelGoddTuring, args.gramsGumber, args.sent, args.idx, args.topK, os.path.join(args.out_dir, "modelGoddTuring.txt"))
        elif args.func == "spearman_analysis":
            analysis_with_spearman_rank_correlation(vocabSmoothLaplace.uniVocab,
                    vocabSmoothLaplace.ProbsVocabs[args.grams_number],
                    vocabSmoothHeldOut.ProbsVocabs[args.grams_number],
                    args.grams_number,
                    os.path.join(args.out_dir, "Laplace-HeldOut.txt"))
            analysis_with_spearman_rank_correlation(vocabSmoothLaplace.uniVocab,
                    vocabSmoothLaplace.ProbsVocabs[args.grams_number],
                    modelCrossValid.ProbsVocabs[args.grams_number],
                    args.grams_number,
                    os.path.join(args.out_dir, "Laplace-CrossValid.txt"))
            analysis_with_spearman_rank_correlation(vocabSmoothLaplace.uniVocab,
                    vocabSmoothLaplace.ProbsVocabs[args.grams_number],
                    vocabSmoothGoodTuring.ProbsVocabs[args.grams_number],
                    args.grams_number,
                    os.path.join(args.out_dir, "Laplace-GoodTuring.txt"))
            analysis_with_spearman_rank_correlation(vocabSmoothLaplace.uniVocab,
                    vocabSmoothHeldOut.ProbsVocabs[args.grams_number],
                    modelCrossValid.ProbsVocabs[args.grams_number],
                    args.grams_number,
                    os.path.join(args.out_dir, "HeldOut-CrossValid.txt"))
            analysis_with_spearman_rank_correlation(vocabSmoothLaplace.uniVocab,
                    vocabSmoothHeldOut.ProbsVocabs[args.grams_number],
                    vocabSmoothGoodTuring.ProbsVocabs[args.grams_number],
                    args.grams_number,
                    os.path.join(args.out_dir, "HeldOut-GoodTuring.txt"))
            analysis_with_spearman_rank_correlation(vocabSmoothLaplace.uniVocab,
                    modelCrossValid.ProbsVocabs[args.grams_number],
                    vocabSmoothGoodTuring.ProbsVocabs[args.grams_number],
                    args.grams_number,
                    os.path.join(args.out_dir, "CrossValid-GoodTuring.txt"))
        else:
            raise IOError
