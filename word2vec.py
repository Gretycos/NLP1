#!/usr/bin/env python

import numpy as np
import random

from utils.gradcheck import gradcheck_naive, grad_tests_softmax, grad_tests_negsamp
from utils.utils import normalizeRows, softmax


def sigmoid(x):
    """
    Compute the sigmoid function for the input here.
    Arguments:
    x -- A scalar or numpy array.
    Return:
    s -- sigmoid(x)
    """

    ### YOUR CODE HERE (~1 Line)
    s = 1 / (1 + np.exp(-x))
    ### END YOUR CODE

    return s


def naiveSoftmaxLossAndGradient(
    centerWordVec,
    outsideWordIdx,
    outsideVectors,
    dataset
):
    """ Naive Softmax loss & gradient function for word2vec models

    Implement the naive softmax loss and gradients between a center word's 
    embedding and an outside word's embedding. This will be the building block
    for our word2vec models.

    Arguments:
    centerWordVec -- numpy ndarray, center word's embedding
                    in shape (word vector length, )
                    (v_c in the pdf handout)
    outsideWordIdx -- integer, the index of the outside word
                    (o of u_o in the pdf handout)
    outsideVectors -- outside vectors is
                    in shape (num words in vocab, word vector length) 
                    for all words in vocab (U in the pdf handout)
    dataset -- needed for negative sampling, unused here.

    Return:
    loss -- naive softmax loss
    gradCenterVec -- the gradient with respect to the center word vector
                     in shape (word vector length, )
                     (dJ / dv_c in the pdf handout)
    gradOutsideVecs -- the gradient with respect to all the outside word vectors
                    in shape (num words in vocab, word vector length) 
                    (dJ / dU)
    """

    ### YOUR CODE HERE (~6-8 Lines)

    ### Please use the provided softmax function (imported earlier in this file)
    ### This numerically stable implementation helps you avoid issues pertaining
    ### to integer overflow.
    x = np.dot(outsideVectors, centerWordVec) # centerV:(词向量长,) outsideVs:(词数,词向量长) x:(词数,)
    p = softmax(x) # p:(词数,)
    loss = - np.log(p[outsideWordIdx])
    gradCenterVec = - outsideVectors[outsideWordIdx] + np.dot(outsideVectors.T, p) # (词向量长,)
    p[outsideWordIdx] = p[outsideWordIdx] - 1
    gradOutsideVecs = np.dot(p[:,None], centerWordVec.T[None,:]) # （词数, 词向量长） 向量之间的点积需要增加一个空维
    ### END YOUR CODE


    return loss, gradCenterVec, gradOutsideVecs # gradCenterVec:(词向量长,) gradOutsideVecs:(词数,词向量长)


def getNegativeSamples(outsideWordIdx, dataset, K):
    """ Samples K indexes which are not the outsideWordIdx """

    negSampleWordIndices = [None] * K
    for k in range(K):
        newidx = dataset.sampleTokenIdx()
        while newidx == outsideWordIdx:
            newidx = dataset.sampleTokenIdx()
        negSampleWordIndices[k] = newidx
    return negSampleWordIndices


def negSamplingLossAndGradient(
    centerWordVec,
    outsideWordIdx,
    outsideVectors,
    dataset,
    K=10
):
    """ Negative sampling loss function for word2vec models

    Implement the negative sampling loss and gradients for a centerWordVec
    and a outsideWordIdx word vector as a building block for word2vec
    models. K is the number of negative samples to take.

    Note: The same word may be negatively sampled multiple times. For
    example if an outside word is sampled twice, you shall have to
    double count the gradient with respect to this word. Thrice if
    it was sampled three times, and so forth.

    Arguments/Return Specifications: same as naiveSoftmaxLossAndGradient
    """

    # Negative sampling of words is done for you. Do not modify this if you
    # wish to match the autograder and receive points!
    negSampleWordIndices = getNegativeSamples(outsideWordIdx, dataset, K) # 索引的索引列表:[neg1,neg2,...,negk]
    indices = [outsideWordIdx] + negSampleWordIndices # 索引的索引列表:[u_o,neg1,neg2,...,negk]

    ### YOUR CODE HERE (~10 Lines)
    ### Please use your implementation of sigmoid in here.
    gradOutsideVecs = np.zeros(outsideVectors.shape) # (词数,词向量长)

    u_o = outsideVectors[outsideWordIdx] # u:(词向量长,)
    sigma = sigmoid(np.dot(u_o, centerWordVec)) # 实数
    loss = -np.log(sigma) # J
    gradCenterVec = u_o*(sigma-1) # dJ / dv_c
    gradOutsideVecs[outsideWordIdx] = centerWordVec*(sigma-1) # dJ/dU第一种情况:dJ / du_o  centerWordVec:(词向量长,)
    # k个负采样的词
    for k in range(K):
        negSampleIdx = negSampleWordIndices[k] # 负采样出的词下标
        u_k = outsideVectors[negSampleIdx] # u:(词向量长,)
        sigma = sigmoid(-np.dot(u_k,centerWordVec)) # 实数
        loss -= np.log(sigma) # J
        gradCenterVec -= u_k*(sigma-1) # dJ / dv_c
        gradOutsideVecs[negSampleIdx] -= centerWordVec*(sigma-1) # dJ/dU第二种情况:dJ / du_k  centerWordVec:(词向量长,)
    ### END YOUR CODE

    return loss, gradCenterVec, gradOutsideVecs # gradCenterVec:(词向量长,) gradOutsideVecs:(词数,词向量长)


def skipgram(currentCenterWord, windowSize, outsideWords, word2Ind,
             centerWordVectors, outsideVectors, dataset,
             word2vecLossAndGradient=naiveSoftmaxLossAndGradient):
    """ Skip-gram model in word2vec

    Implement the skip-gram model in this function.

    Arguments:
    currentCenterWord -- a string of the current center word
    windowSize -- integer, context window size
    outsideWords -- list of no more than 2*windowSize strings, the outside words
    word2Ind -- a dictionary that maps words to their indices in
              the word vector list
    centerWordVectors -- center word vectors (as rows) is in shape 
                        (num words in vocab, word vector length) 
                        for all words in vocab (V in pdf handout)
    outsideVectors -- outside vectors is in shape 
                        (num words in vocab, word vector length) 
                        for all words in vocab (U in the pdf handout)
    word2vecLossAndGradient -- the loss and gradient function for
                               a prediction vector given the outsideWordIdx
                               word vectors, could be one of the two
                               loss functions you implemented above.

    Return:
    loss -- the loss function value for the skip-gram model
            (J in the pdf handout)
    gradCenterVec -- the gradient with respect to the center word vector
                     in shape (word vector length, )
                     (dJ / dv_c in the pdf handout)
    gradOutsideVecs -- the gradient with respect to all the outside word vectors
                    in shape (num words in vocab, word vector length) 
                    (dJ / dU)
    """

    loss = 0.0
    gradCenterVecs = np.zeros(centerWordVectors.shape)
    gradOutsideVectors = np.zeros(outsideVectors.shape)

    ### YOUR CODE HERE (~8 Lines)
    centerWordIdx = word2Ind[currentCenterWord] # 中心词下标
    centerWordVec = centerWordVectors[centerWordIdx] # v_c
    for outsideWord in outsideWords:
        outsideWordIdx = word2Ind[outsideWord]
        # dloss:实数 dgradCenterVecs:(词向量长,)  dgradOutsideVectors:(词数,词向量长)
        dloss, dgradCenterVecs, dgradOutsideVectors = word2vecLossAndGradient(
            centerWordVec=centerWordVec,outsideWordIdx=outsideWordIdx,outsideVectors=outsideVectors,dataset=dataset)
        loss += dloss
        gradCenterVecs[centerWordIdx] += dgradCenterVecs
        gradOutsideVectors += dgradOutsideVectors
    ### END YOUR CODE
    
    return loss, gradCenterVecs, gradOutsideVectors # gradCenterVecs:(词数,词向量长) gradOutsideVectors:(词数,词向量长)


#############################################
# Testing functions below. DO NOT MODIFY!   #
#############################################

def word2vec_sgd_wrapper(word2vecModel, word2Ind, wordVectors, dataset, 
                         windowSize,
                         word2vecLossAndGradient=naiveSoftmaxLossAndGradient):
    batchsize = 50
    loss = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0]
    centerWordVectors = wordVectors[:int(N/2),:]
    outsideVectors = wordVectors[int(N/2):,:]
    for i in range(batchsize):
        windowSize1 = random.randint(1, windowSize)
        centerWord, context = dataset.getRandomContext(windowSize1)

        c, gin, gout = word2vecModel(
            centerWord, windowSize1, context, word2Ind, centerWordVectors,
            outsideVectors, dataset, word2vecLossAndGradient
        )
        loss += c / batchsize
        grad[:int(N/2), :] += gin / batchsize
        grad[int(N/2):, :] += gout / batchsize

    return loss, grad


def test_word2vec():
    """ Test the two word2vec implementations, before running on Stanford Sentiment Treebank """
    dataset = type('dummy', (), {})()
    def dummySampleTokenIdx():
        return random.randint(0, 4)

    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0,4)], \
            [tokens[random.randint(0,4)] for i in range(2*C)]
    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalizeRows(np.random.randn(10,3))
    dummy_tokens = dict([("a",0), ("b",1), ("c",2),("d",3),("e",4)])

    print("==== Gradient check for skip-gram with naiveSoftmaxLossAndGradient ====")
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, naiveSoftmaxLossAndGradient),
        dummy_vectors, "naiveSoftmaxLossAndGradient Gradient")
    grad_tests_softmax(skipgram, dummy_tokens, dummy_vectors, dataset)

    print("==== Gradient check for skip-gram with negSamplingLossAndGradient ====")
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, negSamplingLossAndGradient),
        dummy_vectors, "negSamplingLossAndGradient Gradient")

    grad_tests_negsamp(skipgram, dummy_tokens, dummy_vectors, dataset, negSamplingLossAndGradient)


if __name__ == "__main__":
    test_word2vec()

