# coding=utf-8
import time      
import re
import os
import sys
import codecs
import shutil
import matplotlib
import scipy
import nltk
import numpy as np
import matplotlib.pyplot as plt
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from gensim.models import word2vec
from nltk.corpus import PlaintextCorpusReader
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

DATA_PATH='argoLog_pro_finish.txt'
COLOR=['b','g','r','c','m','y','k']
MARKER=['.',',','o','v','^','<','>','8','s','p','*','h','H','+','x','D','d']

# 读取数据
def loadData(data_path):
    sentences=[]
    for sentence in open(data_path):
        sentences.append(sentence)
    sentences=[s.encode('utf-8').split() for s in sentences]

    train1=[]
    for i in range(len(sentences)*1/10,len(sentences)):
        train1.append(sentences[i])
    valid1=[]
    for i in range(0,len(sentences)*1/10):
        valid1.append(sentences[i])

    train2=[]
    for i in range(0,len(sentences)*1/10):
        train2.append(sentences[i])
    for i in range(len(sentences)*2/10,len(sentences)):
        train2.append(sentences[i])
    valid2=[]
    for i in range(len(sentences)*1/10,len(sentences)*2/10):
        valid2.append(sentences[i])

    train3=[]
    for i in range(0,len(sentences)*2/10):
        train3.append(sentences[i])
    for i in range(len(sentences)*3/10,len(sentences)):
        train3.append(sentences[i])
    valid3=[]
    for i in range(len(sentences)*2/10,len(sentences)*3/10):
        valid3.append(sentences[i])

    train4=[]
    for i in range(0,len(sentences)*3/10):
        train4.append(sentences[i])
    for i in range(len(sentences)*4/10,len(sentences)):
        train4.append(sentences[i])
    valid4=[]
    for i in range(len(sentences)*3/10,len(sentences)*4/10):
        valid4.append(sentences[i])

    train5=[]
    for i in range(0,len(sentences)*4/10):
        train5.append(sentences[i])
    for i in range(len(sentences)*5/10,len(sentences)):
        train5.append(sentences[i])
    valid5=[]
    for i in range(len(sentences)*4/10,len(sentences)*5/10):
        valid5.append(sentences[i])

    train6=[]
    for i in range(0,len(sentences)*5/10):
        train6.append(sentences[i])
    for i in range(len(sentences)*6/10,len(sentences)):
        train6.append(sentences[i])
    valid6=[]
    for i in range(len(sentences)*5/10,len(sentences)*6/10):
        valid6.append(sentences[i])

    train7=[]
    for i in range(0,len(sentences)*6/10):
        train7.append(sentences[i])
    for i in range(len(sentences)*7/10,len(sentences)):
        train7.append(sentences[i])
    valid7=[]
    for i in range(len(sentences)*6/10,len(sentences)*7/10):
       valid7.append(sentences[i])

    train8=[]
    for i in range(0,len(sentences)*7/10):
        train8.append(sentences[i])
    for i in range(len(sentences)*8/10,len(sentences)):
        train8.append(sentences[i])
    valid8=[]
    for i in range(len(sentences)*7/10,len(sentences)*8/10):
       valid8.append(sentences[i])

    train9=[]
    for i in range(0,len(sentences)*8/10):
        train9.append(sentences[i])
    for i in range(len(sentences)*9/10,len(sentences)):
        train9.append(sentences[i])
    valid9=[]
    for i in range(len(sentences)*8/10,len(sentences)*9/10):
       valid9.append(sentences[i])

    train10=[]
    for i in range(0,len(sentences)*9/10):
        train10.append(sentences[i])
    valid10=[]
    for i in range(len(sentences)*9/10,len(sentences)):
       valid10.append(sentences[i])

    return sentences,train1,train2,train3,train4,train5,train6,train7,train8,train9,train10,valid1,valid2,valid3,valid4,valid5,valid6,valid7,valid8,valid9,valid10

# 实现 word to vector
def w2v(sentences):
    model=word2vec.Word2Vec(sentences,min_count=1)
    vectors=[]
    for sentence in sentences:
        vector=[0]*len(model.wv[sentence[0]])
        for word in sentence:
    	    vector=vector+model.wv[word]
        vectors.append(vector)
    return vectors
# 贝叶斯之生成字典
def createVocabs(sentences):
    vocabs=set([])
    for sentence in sentences:
        vocabs=vocabs|set(sentence)
    return list(vocabs)

# 贝叶斯之Sentence To Vector（词集模型）
# def Sentence2Vector(vocabs,sentence):
#     vec=[0] * len(vocabs)
#     for word in sentence:
#         if word in vocabs:
#             vec[vocabs.index(word)]+=1
#         else:
#             print("the word :%s is not in my  Vocabulary!"%word)
#     return vec

def sentence2words(sentence):
    return sentence.split(' ')

# 求得所有的单词
def getAllwords(sentences):
    allwords=set()
    for sentence in sentences:
        for word in sentence:
            allwords.add(word)
    return allwords
# word->index和index->word
def getWord2Index2Word(allwords):
    allwords=list(allwords)
    word2indx={}
    indx2word={}
    for i in range(0,len(allwords)):
        indx2word[i]=allwords[i]
        word2indx[allwords[i]]=i
    return word2indx,indx2word
# 求P(yi)
def getPyi(clf):
    pyi=[0]*K
    for i in range(0,len(clf.labels_)):
        for j in range(0,K):
            if(clf.labels_[i]==j):
                pyi[j]+=1
                break
    for i in range(0,len(pyi)):
        pyi[i]=(pyi[i]+0.0)/len(clf.labels_)
    return pyi
# 代码尚可优化
def getPxPy(sentences,clf,allwords,word2indx):
    pxpy=[[0]*len(allwords) for i in range(K)]
    labelssentences=[[] for i in range(K)]
    labelssum=[]
    for i in range(0,len(clf.labels_)):
        for j in range(0,K):
            if(clf.labels_[i]==j):
                labelssentences[j].append(sentences[i])
                break
    for i in range(0,K):
        for sentence in labelssentences[i]:
            for word in sentence:
                pxpy[i][word2indx[word]]+=1
    for i in range(0,K):
        labelssum.append(sum(pxpy[i]))
    for i in range(0,K):
        for j in range(0,len(pxpy[i])):
            pxpy[i][j]=(pxpy[i][j]+0.1)/labelssum[i]
    return pxpy

def Secderivaion(y):
    ff=[]
    # ff.append(0)#第一个值不能求导，所以暂时赋值为0
    for i in range(1,len(y)-1):
        f=(y[i+1]-2*y[i]+y[i-1])
        ff.append(f)
    # ff.append(0)#最后一个值不能求导，所以暂时赋值为0
    return ff

#对第k类的分类器做评估，计算准确率p，召回率r，调和平均值f（参数β分别取0.5，1，1.5）
# y_true, y_pred
# TP = (y_pred==k)*(y_true==k)
# FP = (y_pred==k)*(y_true!=k)
# FN = (y_pred!=k)*(y_true==k)
# TN = (y_pred!=k)*(y_true!=k)
# TP + FP = y_pred==k
# TP + FN = y_true==k
# TPR=recall
# FPR=FP/(FP+TN)
def countTP(y_true,y_pred,k):
    count=[]
    for i in range(0,len(y_true)):
        if (y_true[i]==y_pred[i]==k):
            count.append(i)
    return len(count)
def countFP(y_true,y_pred,k):
    count=[]
    for i in range(0,len(y_true)):
        if (y_true[i]!=k&y_pred[i]==k):
            count.append(i)
    return len(count)
def countTPaddFP(y_true,y_pred,k):
    count=[]
    for i in range(0,len(y_pred)):
        if (y_pred[i]==k):
            count.append(i)
    return len(count)
def countTPaddFN(y_true,y_pred,k):
    count=[]
    for i in range(0,len(y_true)):
        if (y_true[i]==k):
            count.append(i)
    return len(count)
def precision_score(y_true, y_pred,k):
    return float((countTP(y_true,y_pred,k)+1.0)/(countTPaddFP(y_true,y_pred,k)+1.0))
def recall_score(y_true, y_pred,k):
    return float((countTP(y_true,y_pred,k)+1.0)/(countTPaddFN(y_true,y_pred,k)+1.0))
def f05_score(y_true, y_pred,k):
    num = 1.25*precision_score(y_true, y_pred,k)*recall_score(y_true, y_pred,k)
    deno = 0.25*(precision_score(y_true, y_pred,k))+recall_score(y_true, y_pred,k)
    return float(num/deno)
def f1_score(y_true, y_pred,k):
    num = 2.0*precision_score(y_true, y_pred,k)*recall_score(y_true, y_pred,k)
    deno = precision_score(y_true, y_pred,k)+recall_score(y_true, y_pred,k)
    return float(num/deno)
def f15_score(y_true, y_pred,k):
    num = 3.25*precision_score(y_true, y_pred,k)*recall_score(y_true, y_pred,k)
    deno = 2.25*(precision_score(y_true, y_pred,k))+recall_score(y_true, y_pred,k)
    return float(num/deno)

if __name__ == "__main__":
    sentences,data_train1,data_train2,data_train3,data_train4,data_train5,data_train6,data_train7,data_train8,data_train9,data_train10,data_valid1,data_valid2,data_valid3,data_valid4,data_valid5,data_valid6,data_valid7,data_valid8,data_valid9,data_valid10=loadData(DATA_PATH)

    vectors=w2v(sentences)
    print vectors[50]

    K=9
    clf=KMeans(n_clusters=K)
    clf.fit(vectors)
    # print clf.inertia_
    

    # 聚类效果图
    pca=PCA(n_components=2)
    data=pca.fit_transform(vectors)
    fig = plt.figure()
    x={}
    y={}
    for i in range(0,K):
        x[i]=[]
    for i in range(0,K):
        y[i]=[]
    for i in range(0,len(clf.labels_)):
        x[clf.labels_[i]].append(data[i][0])
        y[clf.labels_[i]].append(data[i][1])
    for k in range(0,K):
        plt.scatter(x[k],y[k],c=COLOR[k%len(COLOR)],alpha=1,s=6,marker=MARKER[k%len(MARKER)])   
        # 设置坐标轴的注释
        plt.title('Graph of Clustering')
        # plt.axis([-100,900,-150,350]) 
        plt.xlabel('X')
        plt.ylabel('Y')
        # 设置图例
        plt.legend(labels = ['1', '2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','15','26','27','28','29','30','31',], loc = 'best')
    plt.show()
    print clf.labels_[0:500]


    ##注释代码也有用！注释代码也有用！注释代码也有用！注释代码也有用！

    # # nb分类器分类操作
    # vocabs=createVocabs(sentences)
    # allwords=getAllwords(sentences)
    # pyi=getPyi(clf)
    # # print pyi
    # word2indx,indx2word=getWord2Index2Word(allwords)
    # # print word2indx
    # # print indx2word
    # pxpy=getPxPy(sentences,clf,allwords,word2indx)

    # pk1=[]
    # for i in range(0,len(data_valid1)):
    #     Pk=[]
    #     for k in range(0,K):
    #         multi=1.0
    #         for word in data_valid1[i]:
    #             multi*=pxpy[k][word2indx[word]]
    #         Pk.append(multi*pyi[k])
    #         # print Pk
    #     pk1.append(Pk.index(max(Pk)))

    # pk2=[]
    # for i in range(0,len(data_valid2)):
    #     Pk=[]
    #     for k in range(0,K):
    #         multi=1.0
    #         for word in data_valid2[i]:
    #             multi*=pxpy[k][word2indx[word]]
    #         Pk.append(multi*pyi[k])
    #     pk2.append(Pk.index(max(Pk)))

    # pk3=[]
    # for i in range(0,len(data_valid3)):
    #     Pk=[]
    #     for k in range(0,K):
    #         multi=1.0
    #         for word in data_valid3[i]:
    #             multi*=pxpy[k][word2indx[word]]
    #         Pk.append(multi*pyi[k])
    #     pk3.append(Pk.index(max(Pk)))

    # pk4=[]
    # for i in range(0,len(data_valid4)):
    #     Pk=[]
    #     for k in range(0,K):
    #         multi=1.0
    #         for word in data_valid4[i]:
    #             multi*=pxpy[k][word2indx[word]]
    #         Pk.append(multi*pyi[k])
    #     pk4.append(Pk.index(max(Pk)))

    # pk5=[]
    # for i in range(0,len(data_valid5)):
    #     Pk=[]
    #     for k in range(0,K):
    #         multi=1.0
    #         for word in data_valid5[i]:
    #             multi*=pxpy[k][word2indx[word]]
    #         Pk.append(multi*pyi[k])
    #     pk5.append(Pk.index(max(Pk)))

    # pk6=[]
    # for i in range(0,len(data_valid6)):
    #     Pk=[]
    #     for k in range(0,K):
    #         multi=1.0
    #         for word in data_valid6[i]:
    #             multi*=pxpy[k][word2indx[word]]
    #         Pk.append(multi*pyi[k])
    #     pk6.append(Pk.index(max(Pk)))

    # pk7=[]
    # for i in range(0,len(data_valid7)):
    #     Pk=[]
    #     for k in range(0,K):
    #         multi=1.0
    #         for word in data_valid7[i]:
    #             multi*=pxpy[k][word2indx[word]]
    #         Pk.append(multi*pyi[k])
    #     pk7.append(Pk.index(max(Pk)))

    # pk8=[]
    # for i in range(0,len(data_valid8)):
    #     Pk=[]
    #     for k in range(0,K):
    #         multi=1.0
    #         for word in data_valid8[i]:
    #             multi*=pxpy[k][word2indx[word]]
    #         Pk.append(multi*pyi[k])
    #     pk8.append(Pk.index(max(Pk)))

    # pk9=[]
    # for i in range(0,len(data_valid9)):
    #     Pk=[]
    #     for k in range(0,K):
    #         multi=1.0
    #         for word in data_valid9[i]:
    #             multi*=pxpy[k][word2indx[word]]
    #         Pk.append(multi*pyi[k])
    #     pk9.append(Pk.index(max(Pk)))

    # pk10=[]
    # for i in range(0,len(data_valid10)):
    #     Pk=[]
    #     for k in range(0,K):
    #         multi=1.0
    #         for word in data_valid10[i]:
    #             multi*=pxpy[k][word2indx[word]]
    #         Pk.append(multi*pyi[k])
    #     pk10.append(Pk.index(max(Pk)))

    # # 分类效果图
    # vectors1=w2v(data_valid1)
    # pca=PCA(n_components=2)
    # data=pca.fit_transform(vectors1)
    # fig = plt.figure()
    # x={}
    # y={}
    # for i in range(0,K):
    #     x[i]=[]
    # for i in range(0,K):
    #     y[i]=[]
    # for i in range(0,len(pk1)):
    #     label=pk1[i]
    #     for k in range(0,K):
    #         if(label==k):
    #             x[k].append(data[i][0])
    #             y[k].append(data[i][1])
    #             break
    # for k in range(0,K):
    #     plt.scatter(x[k],y[k],c=COLOR[k%len(COLOR)],alpha=1,s=6,marker=MARKER[k%len(MARKER)])     
    #     # 设置坐标轴的注释
    #     plt.title('Graph of Validating(1)')
    #     # plt.axis([-100,900,-150,350]) 
    #     plt.xlabel('X')
    #     plt.ylabel('Y')
    #     # 设置图例
    #     plt.legend(labels = ['1', '2','3','4','5','6','7','8','9',], loc = 'best')
    # plt.show()

    # vectors2=w2v(data_valid2)
    # pca=PCA(n_components=2)
    # data=pca.fit_transform(vectors2)
    # fig = plt.figure()
    # x={}
    # y={}
    # for i in range(0,K):
    #     x[i]=[]
    # for i in range(0,K):
    #     y[i]=[]
    # for i in range(0,len(pk2)):
    #     label=pk2[i]
    #     for k in range(0,K):
    #         if(label==k):
    #             x[k].append(data[i][0])
    #             y[k].append(data[i][1])
    #             break
    # for k in range(0,K):
    #     plt.scatter(x[k],y[k],c=COLOR[k%len(COLOR)],alpha=1,s=6,marker=MARKER[k%len(MARKER)])     
    #     # 设置坐标轴的注释
    #     plt.title('Graph of Validating(2)')
    #     # plt.axis([-100,900,-150,350]) 
    #     plt.xlabel('X')
    #     plt.ylabel('Y')
    #     # 设置图例
    #     plt.legend(labels = ['1', '2','3','4','5','6','7','8','9',], loc = 'best')
    # plt.show()

    # vectors3=w2v(data_valid3)
    # pca=PCA(n_components=2)
    # data=pca.fit_transform(vectors3)
    # fig = plt.figure()
    # x={}
    # y={}
    # for i in range(0,K):
    #     x[i]=[]
    # for i in range(0,K):
    #     y[i]=[]
    # for i in range(0,len(pk3)):
    #     label=pk3[i]
    #     for k in range(0,K):
    #         if(label==k):
    #             x[k].append(data[i][0])
    #             y[k].append(data[i][1])
    #             break
    # for k in range(0,K):
    #     plt.scatter(x[k],y[k],c=COLOR[k%len(COLOR)],alpha=1,s=6,marker=MARKER[k%len(MARKER)])     
    #     # 设置坐标轴的注释
    #     plt.title('Graph of Validating(3)')
    #     # plt.axis([-100,900,-150,350]) 
    #     plt.xlabel('X')
    #     plt.ylabel('Y')
    #     # 设置图例
    #     plt.legend(labels = ['1', '2','3','4','5','6','7','8','9',], loc = 'best')
    # plt.show()

    # vectors4=w2v(data_valid4)
    # pca=PCA(n_components=2)
    # data=pca.fit_transform(vectors4)
    # fig = plt.figure()
    # x={}
    # y={}
    # for i in range(0,K):
    #     x[i]=[]
    # for i in range(0,K):
    #     y[i]=[]
    # for i in range(0,len(pk4)):
    #     label=pk4[i]
    #     for k in range(0,K):
    #         if(label==k):
    #             x[k].append(data[i][0])
    #             y[k].append(data[i][1])
    #             break
    # for k in range(0,K):
    #     plt.scatter(x[k],y[k],c=COLOR[k%len(COLOR)],alpha=1,s=6,marker=MARKER[k%len(MARKER)])     
    #     # 设置坐标轴的注释
    #     plt.title('Graph of Validating(4)')
    #     # plt.axis([-100,900,-150,350]) 
    #     plt.xlabel('X')
    #     plt.ylabel('Y')
    #     # 设置图例
    #     plt.legend(labels = ['1', '2','3','4','5','6','7','8','9',], loc = 'best')
    # plt.show()

    # vectors5=w2v(data_valid5)
    # pca=PCA(n_components=2)
    # data=pca.fit_transform(vectors5)
    # fig = plt.figure()
    # x={}
    # y={}
    # for i in range(0,K):
    #     x[i]=[]
    # for i in range(0,K):
    #     y[i]=[]
    # for i in range(0,len(pk5)):
    #     label=pk5[i]
    #     for k in range(0,K):
    #         if(label==k):
    #             x[k].append(data[i][0])
    #             y[k].append(data[i][1])
    #             break
    # for k in range(0,K):
    #     plt.scatter(x[k],y[k],c=COLOR[k%len(COLOR)],alpha=1,s=6,marker=MARKER[k%len(MARKER)])     
    #     # 设置坐标轴的注释
    #     plt.title('Graph of Validating(5)')
    #     # plt.axis([-100,900,-150,350]) 
    #     plt.xlabel('X')
    #     plt.ylabel('Y')
    #     # 设置图例
    #     plt.legend(labels = ['1', '2','3','4','5','6','7','8','9',], loc = 'best')
    # plt.show()

    # vectors6=w2v(data_valid6)
    # pca=PCA(n_components=2)
    # data=pca.fit_transform(vectors6)
    # fig = plt.figure()
    # x={}
    # y={}
    # for i in range(0,K):
    #     x[i]=[]
    # for i in range(0,K):
    #     y[i]=[]
    # for i in range(0,len(pk6)):
    #     label=pk6[i]
    #     for k in range(0,K):
    #         if(label==k):
    #             x[k].append(data[i][0])
    #             y[k].append(data[i][1])
    #             break
    # for k in range(0,K):
    #     plt.scatter(x[k],y[k],c=COLOR[k%len(COLOR)],alpha=1,s=6,marker=MARKER[k%len(MARKER)])     
    #     # 设置坐标轴的注释
    #     plt.title('Graph of Validating(6)')
    #     # plt.axis([-100,900,-150,350]) 
    #     plt.xlabel('X')
    #     plt.ylabel('Y')
    #     # 设置图例
    #     plt.legend(labels = ['1', '2','3','4','5','6','7','8','9',], loc = 'best')
    # plt.show()

    # vectors7=w2v(data_valid7)
    # pca=PCA(n_components=2)
    # data=pca.fit_transform(vectors7)
    # fig = plt.figure()
    # x={}
    # y={}
    # for i in range(0,K):
    #     x[i]=[]
    # for i in range(0,K):
    #     y[i]=[]
    # for i in range(0,len(pk7)):
    #     label=pk7[i]
    #     for k in range(0,K):
    #         if(label==k):
    #             x[k].append(data[i][0])
    #             y[k].append(data[i][1])
    #             break
    # for k in range(0,K):
    #     plt.scatter(x[k],y[k],c=COLOR[k%len(COLOR)],alpha=1,s=6,marker=MARKER[k%len(MARKER)])     
    #     # 设置坐标轴的注释
    #     plt.title('Graph of Validating(7)')
    #     # plt.axis([-100,900,-150,350]) 
    #     plt.xlabel('X')
    #     plt.ylabel('Y')
    #     # 设置图例
    #     plt.legend(labels = ['1', '2','3','4','5','6','7','8','9',], loc = 'best')
    # plt.show()

    # vectors8=w2v(data_valid8)
    # pca=PCA(n_components=2)
    # data=pca.fit_transform(vectors8)
    # fig = plt.figure()
    # x={}
    # y={}
    # for i in range(0,K):
    #     x[i]=[]
    # for i in range(0,K):
    #     y[i]=[]
    # for i in range(0,len(pk8)):
    #     label=pk8[i]
    #     for k in range(0,K):
    #         if(label==k):
    #             x[k].append(data[i][0])
    #             y[k].append(data[i][1])
    #             break
    # for k in range(0,K):
    #     plt.scatter(x[k],y[k],c=COLOR[k%len(COLOR)],alpha=1,s=6,marker=MARKER[k%len(MARKER)])     
    #     # 设置坐标轴的注释
    #     plt.title('Graph of Validating(8)')
    #     # plt.axis([-100,900,-150,350]) 
    #     plt.xlabel('X')
    #     plt.ylabel('Y')
    #     # 设置图例
    #     plt.legend(labels = ['1', '2','3','4','5','6','7','8','9',], loc = 'best')
    # plt.show()

    # vectors9=w2v(data_valid9)
    # pca=PCA(n_components=2)
    # data=pca.fit_transform(vectors9)
    # fig = plt.figure()
    # x={}
    # y={}
    # for i in range(0,K):
    #     x[i]=[]
    # for i in range(0,K):
    #     y[i]=[]
    # for i in range(0,len(pk9)):
    #     label=pk9[i]
    #     for k in range(0,K):
    #         if(label==k):
    #             x[k].append(data[i][0])
    #             y[k].append(data[i][1])
    #             break
    # for k in range(0,K):
    #     plt.scatter(x[k],y[k],c=COLOR[k%len(COLOR)],alpha=1,s=6,marker=MARKER[k%len(MARKER)])     
    #     # 设置坐标轴的注释
    #     plt.title('Graph of Validating(9)')
    #     # plt.axis([-100,900,-150,350]) 
    #     plt.xlabel('X')
    #     plt.ylabel('Y')
    #     # 设置图例
    #     plt.legend(labels = ['1', '2','3','4','5','6','7','8','9',], loc = 'best')
    # plt.show()

    # vectors10=w2v(data_valid10)
    # pca=PCA(n_components=2)
    # data=pca.fit_transform(vectors10)
    # fig = plt.figure()
    # x={}
    # y={}
    # for i in range(0,K):
    #     x[i]=[]
    # for i in range(0,K):
    #     y[i]=[]
    # for i in range(0,len(pk10)):
    #     label=pk10[i]
    #     for k in range(0,K):
    #         if(label==k):
    #             x[k].append(data[i][0])
    #             y[k].append(data[i][1])
    #             break
    # for k in range(0,K):
    #     plt.scatter(x[k],y[k],c=COLOR[k%len(COLOR)],alpha=1,s=6,marker=MARKER[k%len(MARKER)])     
    #     # 设置坐标轴的注释
    #     plt.title('Graph of Validating(10)')
    #     # plt.axis([-100,900,-150,350]) 
    #     plt.xlabel('X')
    #     plt.ylabel('Y')
    #     # 设置图例
    #     plt.legend(labels = ['1', '2','3','4','5','6','7','8','9',], loc = 'best')
    # plt.show()

    # # 计算每个类的准确率，召回率，调和平均数（参数β分别取0.5，1，1.5）
    # print '---------------------------------------------------实验1---------------------------------------------------'
    # pk1=np.array(pk1)
    # clf.labels_=np.array(clf.labels_)
    # print pk1
    # print clf.labels_[0:int(len(sentences)*1/10)]
    # for i in range(0,K):
    #     k=i
    #     # print i
    #     print '--------------------------------------'
    #     # print 'precision:'
    #     print precision_score(clf.labels_[0:int(len(sentences)*1/10)],pk1,k)
    #     # print 'recall:'
    #     print recall_score(clf.labels_[0:int(len(sentences)*1/10)],pk1,k)
    #     # print 'f0.5:'
    #     print f05_score(clf.labels_[0:int(len(sentences)*1/10)],pk1,k)
    #     # print 'f1:'
    #     print f1_score(clf.labels_[0:int(len(sentences)*1/10)],pk1,k)
    #     # print 'f1.5'
    #     print f15_score(clf.labels_[0:int(len(sentences)*1/10)],pk1,k)
    #     print '--------------------------------------'
    # print '---------------------------------------------------实验2---------------------------------------------------'
    # pk2=np.array(pk2)
    # clf.labels_=np.array(clf.labels_)
    # for i in range(0,K):
    #     k=i
    #     # print i
    #     print '--------------------------------------'
    #     # print 'precision:'
    #     print precision_score(clf.labels_[int(len(sentences)*1/10):int(len(sentences)*2/10)],pk2,k)
    #     # print 'recall:'
    #     print recall_score(clf.labels_[int(len(sentences)*1/10):int(len(sentences)*2/10)],pk2,k)
    #     # print 'f0.5:'
    #     print f05_score(clf.labels_[int(len(sentences)*1/10):int(len(sentences)*2/10)],pk2,k)
    #     # print 'f1:'
    #     print f1_score(clf.labels_[int(len(sentences)*1/10):int(len(sentences)*2/10)],pk2,k)
    #     # print 'f1.5'
    #     print f15_score(clf.labels_[int(len(sentences)*1/10):int(len(sentences)*2/10)],pk2,k)
    #     print '--------------------------------------'
    # print '---------------------------------------------------实验3---------------------------------------------------'
    # pk3=np.array(pk3)
    # clf.labels_=np.array(clf.labels_)
    # for i in range(0,K):
    #     k=i
    #     # print i
    #     print '--------------------------------------'
    #     # print 'precision:'
    #     print precision_score(clf.labels_[int(len(sentences)*2/10):int(len(sentences)*3/10)],pk3,k)
    #     # print 'recall:'
    #     print recall_score(clf.labels_[int(len(sentences)*2/10):int(len(sentences)*3/10)],pk3,k)
    #     # print 'f0.5:'
    #     print f05_score(clf.labels_[int(len(sentences)*2/10):int(len(sentences)*3/10)],pk3,k)
    #     # print 'f1:'
    #     print f1_score(clf.labels_[int(len(sentences)*2/10):int(len(sentences)*3/10)],pk3,k)
    #     # print 'f1.5'
    #     print f15_score(clf.labels_[int(len(sentences)*2/10):int(len(sentences)*3/10)],pk3,k)
    #     print '--------------------------------------'
    # print '---------------------------------------------------实验4---------------------------------------------------'
    # pk4=np.array(pk4)
    # clf.labels_=np.array(clf.labels_)
    # for i in range(0,K):
    #     k=i
    #     # print i
    #     print '--------------------------------------'
    #     # print 'precision:'
    #     print precision_score(clf.labels_[int(len(sentences)*3/10):int(len(sentences)*4/10)],pk4,k)
    #     # print 'recall:'
    #     print recall_score(clf.labels_[int(len(sentences)*3/10):int(len(sentences)*4/10)],pk4,k)
    #     # print 'f0.5:'
    #     print f05_score(clf.labels_[int(len(sentences)*3/10):int(len(sentences)*4/10)],pk4,k)
    #     # print 'f1:'
    #     print f1_score(clf.labels_[int(len(sentences)*3/10):int(len(sentences)*4/10)],pk4,k)
    #     # print 'f1.5'
    #     print f15_score(clf.labels_[int(len(sentences)*3/10):int(len(sentences)*4/10)],pk4,k)
    #     print '--------------------------------------'
    # print '---------------------------------------------------实验5---------------------------------------------------'
    # pk5=np.array(pk5)
    # clf.labels_=np.array(clf.labels_)
    # for i in range(0,K):
    #     k=i
    #     # print i
    #     print '--------------------------------------'
    #     # print 'precision:'
    #     print precision_score(clf.labels_[int(len(sentences)*4/10):int(len(sentences)*5/10)],pk5,k)
    #     # print 'recall:'
    #     print recall_score(clf.labels_[int(len(sentences)*4/10):int(len(sentences)*5/10)],pk5,k)
    #     # print 'f0.5:'
    #     print f05_score(clf.labels_[int(len(sentences)*4/10):int(len(sentences)*5/10)],pk5,k)
    #     # print 'f1:'
    #     print f1_score(clf.labels_[int(len(sentences)*4/10):int(len(sentences)*5/10)],pk5,k)
    #     # print 'f1.5'
    #     print f15_score(clf.labels_[int(len(sentences)*4/10):int(len(sentences)*5/10)],pk5,k)
    #     print '--------------------------------------'
    # print '---------------------------------------------------实验6---------------------------------------------------'
    # pk6=np.array(pk6)
    # clf.labels_=np.array(clf.labels_)
    # for i in range(0,K):
    #     k=i
    #     # print i
    #     print '--------------------------------------'
    #     # print 'precision:'
    #     print precision_score(clf.labels_[int(len(sentences)*5/10):int(len(sentences)*6/10)],pk6,k)
    #     # print 'recall:'
    #     print recall_score(clf.labels_[int(len(sentences)*5/10):int(len(sentences)*6/10)],pk6,k)
    #     # print 'f0.5:'
    #     print f05_score(clf.labels_[int(len(sentences)*5/10):int(len(sentences)*6/10)],pk6,k)
    #     # print 'f1:'
    #     print f1_score(clf.labels_[int(len(sentences)*5/10):int(len(sentences)*6/10)],pk6,k)
    #     # print 'f1.5'
    #     print f15_score(clf.labels_[int(len(sentences)*5/10):int(len(sentences)*6/10)],pk6,k)
    #     print '--------------------------------------'
    # print '---------------------------------------------------实验7---------------------------------------------------'
    # pk7=np.array(pk7)
    # clf.labels_=np.array(clf.labels_)
    # for i in range(0,K):
    #     k=i
    #     # print i
    #     print '--------------------------------------'
    #     # print 'precision:'
    #     print precision_score(clf.labels_[int(len(sentences)*6/10):int(len(sentences)*7/10)],pk7,k)
    #     # print 'recall:'
    #     print recall_score(clf.labels_[int(len(sentences)*6/10):int(len(sentences)*7/10)],pk7,k)
    #     # print 'f0.5:'
    #     print f05_score(clf.labels_[int(len(sentences)*6/10):int(len(sentences)*7/10)],pk7,k)
    #     # print 'f1:'
    #     print f1_score(clf.labels_[int(len(sentences)*6/10):int(len(sentences)*7/10)],pk7,k)
    #     # print 'f1.5'
    #     print f15_score(clf.labels_[int(len(sentences)*6/10):int(len(sentences)*7/10)],pk7,k)
    #     print '--------------------------------------'
    # print '---------------------------------------------------实验8---------------------------------------------------'
    # pk8=np.array(pk8)
    # clf.labels_=np.array(clf.labels_)
    # for i in range(0,K):
    #     k=i
    #     # print i
    #     print '--------------------------------------'
    #     # print 'precision:'
    #     print precision_score(clf.labels_[int(len(sentences)*7/10):int(len(sentences)*8/10)],pk8,k)
    #     # print 'recall:'
    #     print recall_score(clf.labels_[int(len(sentences)*7/10):int(len(sentences)*8/10)],pk8,k)
    #     # print 'f0.5:'
    #     print f05_score(clf.labels_[int(len(sentences)*7/10):int(len(sentences)*8/10)],pk8,k)
    #     # print 'f1:'
    #     print f1_score(clf.labels_[int(len(sentences)*7/10):int(len(sentences)*8/10)],pk8,k)
    #     # print 'f1.5'
    #     print f15_score(clf.labels_[int(len(sentences)*7/10):int(len(sentences)*8/10)],pk8,k)
    #     print '--------------------------------------'
    # print '---------------------------------------------------实验9---------------------------------------------------'
    # pk9=np.array(pk9)
    # clf.labels_=np.array(clf.labels_)
    # for i in range(0,K):
    #     k=i
    #     # print i
    #     print '--------------------------------------'
    #     # print 'precision:'
    #     print precision_score(clf.labels_[int(len(sentences)*8/10):int(len(sentences)*9/10)],pk9,k)
    #     # print 'recall:'
    #     print recall_score(clf.labels_[int(len(sentences)*8/10):int(len(sentences)*9/10)],pk9,k)
    #     # print 'f0.5:'
    #     print f05_score(clf.labels_[int(len(sentences)*8/10):int(len(sentences)*9/10)],pk9,k)
    #     # print 'f1:'
    #     print f1_score(clf.labels_[int(len(sentences)*8/10):int(len(sentences)*9/10)],pk9,k)
    #     # print 'f1.5'
    #     print f15_score(clf.labels_[int(len(sentences)*8/10):int(len(sentences)*9/10)],pk9,k)
    #     print '--------------------------------------'
    # print '---------------------------------------------------实验10---------------------------------------------------'
    # pk10=np.array(pk10)
    # clf.labels_=np.array(clf.labels_)
    # for i in range(0,K):
    #     k=i
    #     # print i
    #     print '--------------------------------------'
    #     # print 'precision:'
    #     print precision_score(clf.labels_[int(len(sentences)*9/10):int(len(sentences))],pk10,k)
    #     # print 'recall:'
    #     print recall_score(clf.labels_[int(len(sentences)*9/10):int(len(sentences))],pk10,k)
    #     # print 'f0.5:'
    #     print f05_score(clf.labels_[int(len(sentences)*9/10):int(len(sentences))],pk10,k)
    #     # print 'f1:'
    #     print f1_score(clf.labels_[int(len(sentences)*9/10):int(len(sentences))],pk10,k)
    #     # print 'f1.5'
    #     print f15_score(clf.labels_[int(len(sentences)*9/10):int(len(sentences))],pk10,k)
    #     print '--------------------------------------'


    # # 1次实验
    # m=[]
    # n=[]
    # for i in range(11,35):
    #     print(i)
    #     K=i
    #     clf=KMeans(n_clusters=K)
    #     clf.fit(vectors)
    #     #用来评估簇的个数是否合适，距离越小说明簇分的越好，选取临界点的簇个数
    #     # print(clf.inertia_)
    #     m.append(K)
    #     n.append(clf.inertia_)
    # # # 坐标图
    # ff=Secderivaion(n)
    # x = m
    # y = n
    # plt.plot(x,y)
    # # 设置坐标轴的注释
    # plt.xlabel('Clustering Number')
    # plt.ylabel('Average Distance')
    # del m[len(m)-1]
    # del m[0]
    # x = m
    # y = ff
    # plt.plot(x,y)
    # group_labels = m
    # # 设置图例
    # plt.legend(labels = ['a', 'b'], loc = 'best')
    # plt.xticks(x, group_labels, rotation=0)
    # plt.grid()
    # plt.show()

    # # 10次实验求最佳
    # for i in range(0,11):
    #     m=[]
    #     n=[]
    #     for i in range(5,31):
    #         print(i)
    #         K=i
    #         clf=KMeans(n_clusters=K)
    #         # print clf
    #         clf.fit(vectors)
    #         #用来评估簇的个数是否合适，距离越小说明簇分的越好，选取临界点的簇个数
    #         # print(clf.inertia_)
    #         m.append(K)
    #         n.append(clf.inertia_)
    #     # # 坐标图
    #     ff=Secderivaion(n)
    #     # 子图形式显示
    #     fig,ax = plt.subplots()
    #     # 设置坐标轴的注释
    #     plt.xlabel('Clustering Number')
    #     plt.ylabel('Changing Rate of Average Distance')
    #     # 设置图例
    #     plt.legend(labels = ['1', '2','3','4','5','6','7','8','9','10','11',], loc = 'best')
    #     del m[len(m)-1]
    #     del m[0]
    #     x = m
    #     y = ff
    #     group_labels = m
    #     plt.plot(x,y)
    # plt.xticks(x, group_labels, rotation=0)
    # plt.grid()
    # plt.show()

'''
可改进问题：

>>> word2vec词向量度量转句子向量度量问题
        目前处理的方法只是简单进行句子间向量相加

>>> 代码重构

>>> 处理更多的数据

'''