# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 14:37:34 2019

@author: duany
"""

import os

import pandas as pd

def readcolname(filepath):
    file=open(filepath)
    strfile=file.read()
    file.close()
    linelist=strfile.split('\n')
    
    colstr=linelist[5]
    
    cols=colstr.split('\t')
    #print(cols)
    return cols

#readcolname('F:/OneDrive/学习/2019-8/pt/data/2015/AAXJ.txt')    
def readfilepath(REPORT_PATH):

    filelist=[[],[],[]]
    fullpathlist=[[],[],[]]
    
    for i in range(len(REPORT_PATH)):
        for dirpath, dirnames, filenames in os.walk(REPORT_PATH[i]):
            for file in filenames:
                filelist[i].append(file)
                fullpath = os.path.join(dirpath, file)
                fullpathlist[i].append(fullpath)

    return fullpathlist,filelist

def checkcolname(fullpathlist):
    colname=readcolname('./data/2015/AAXJ.txt')
    differentcol=0
    for i in range(len(fullpathlist)):
        for j in range(len(fullpathlist[i])):
            if colname!=readcolname(fullpathlist[i][j]):
                differentcol+=1
    return differentcol

def countNA(fullpathlist):
    sumlist = pd.read_csv('./data/2015/AAXJ.txt', sep="\t",skiprows=5).isna().sum()
    for i in range(len(fullpathlist)):
        for j in range(len(fullpathlist[i])):
            sumlist+=(pd.read_csv(fullpathlist[i][j], sep="\t",skiprows=5)).isna().sum()
    return sumlist

def findoverlap(filelist):
    maxlist = []
    maplist = {'init': -1 }
    for i in range(len(filelist)):
        for j in range(len(filelist[i])):
            if maplist.get(filelist[i][j]):
                maplist[filelist[i][j]]+=1
                if maplist[filelist[i][j]]==3:
                    maxlist.append(filelist[i][j])
            else:
                maplist[filelist[i][j]]=1
    return maplist, maxlist

fullpathlist,filelist=readfilepath(['./data/2015/','./data/2016/','./data/2017/'])

print("Number of different col = ",checkcolname(fullpathlist))

print("Number of NA\n",countNA(fullpathlist))

maplist,maxlist=findoverlap(filelist)


