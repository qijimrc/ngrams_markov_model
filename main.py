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

    import ipdb
    ipdb.set_trace()
    vocab = Vocabulary(gramsNumber, corpusTrainDev)
    if smoothStrategy == "laplace":
        vocab.tune_with_Laplace_smoothing(BLaplace)
    elif smoothStrategy == "held_out":
        vocab.tune_with_held_out_smoothing(corpusDev)
    elif smoothStrategy == "cross_valid":
        vocab.tune_with_cross_val_smoothing(corpusDev)
    elif smoothStrategy == "good_turing":
        vocab.tune_with_good_turing_smoothing()
    else:
        pass
    return vocab

def predidct_file(model, gramsGumber, testFile, saveFile):
    # predict
    rt = []
    with open(testFile, "r") as f:
        for line in f:
            sentProb, sentPP = model.predict_sent_prob(line, gramsGumber)
            print(sentProb, sentPP)
            rt.append((sentProb, sentPP))
    # save results
    with open(saveFile, "w") as f:
        for sentProb, sentPP in rt:
            f.write(sentProb+"\t"+sentPP+"\n")

def predict_sent(model, gramsNumber, sent, idx, topK, saveFile):
    import ipdb
    ipdb.set_trace()
    rt = model.predict_topk_with_index(sent, gramsNumber, idx, topK)
    print(rt)
    # save results
    with open(saveFile, "w") as f:
        f.write("sent: " + sent + "\n")
        f.write("top-k probs: ")
        f.write(" ".join(["({},{})".format(tok, str(prob)) for tok,prob in
                          rt[0]])+"\n")
        f.write("Current token: {}, with probs: {}, and ranking index:\
                {}\n".format(rt[1][0], str(rt[1][1]), str(rt[1][2])))

def analysis_with_spearman_rank_correlation(uniVocab, ngramsProbsVocabsA,
                                            ngramsProbsVocabsB, gramsNumber,
                                            saveFile):
    rt = an_rank_correlation_annalysis(uniVocab, ngramsProbsVocabsA, ngramsProbsVocabsB, gramsNumber)
    with open(saveFile, "w") as f:
        f.write("The correlation result between two methods is {}".format(str(rt)))
    return rt




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--func", type=str, default="pred_sent", help="which function will be used.")
    parser.add_argument("--train_file", type=str, default="data/corpus-new/s3/train.txt")
    parser.add_argument("--dev_file", type=str, default="data/corpus-new/s3/valid.txt")
    parser.add_argument("--test_file", type=str, default="data/corpus-new/s3/test.txt")
    parser.add_argument("--sent", type=str, default="原定 计划 是 早晨 在 一 座 山 上 吃 午饭")
    parser.add_argument("--idx", type=int, default=1)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--vocab_dir", type=str, default="outputs/vocabs")
    parser.add_argument("--out_dir", type=str, default="outputs/results")
    parser.add_argument("--grams_number", type=int, default=2)
    parser.add_argument("--smooth_strategy", type=str, default="laplace")
    parser.add_argument("--B_laplace", type=int, default=1)
    parser.add_argument("--analysis_model1", type=str, default="laplace")
    parser.add_argument("--analysis_model2", type=str, default="heldOut")
    args = parser.parse_args()

    vocab = train(args.train_file,
                args.dev_file,
                args.grams_number,
                args.smooth_strategy,
                args.B_laplace,
                args.vocab_dir)
    model = MarkovModel(vocab)

    if args.func == "pred_file":
        predidct_file(model, args.grams_number, args.test_file,
                      os.path.join(args.out_dir,
                                   "test_file_results."+args.smooth_strategy+".txt"))
    elif args.func == "pred_sent":
        predict_sent(model, args.grams_number, args.sent, args.idx, args.topk,
                     os.path.join(args.out_dir, "test_sent_results"+args.smooth_strategy+".txt"))
    elif args.func == "spearman_analysis":
        vocab1 = train(args.train_file,
                      args.dev_file,
                      args.analysis_model1,
                      args.B_laplace,
                      args.vocab_dir)
        vocab2 = train(args.train_file,
                      args.dev_file,
                      args.analysis_model2,
                      args.B_laplace,
                      args.vocab_dir)
        analysis_with_spearman_rank_correlation(vocabSmoothLaplace.uniVocab,
                vocab1.ProbsVocabs[args.grams_number],
                vocab2.ProbsVocabs[args.grams_number],
                args.grams_number,
                os.path.join(args.out_dir,
                             "analysis_results."+args.analysis_model1+"-"+args.analysis_model2+".txt"))
    else:
        raise IOError
