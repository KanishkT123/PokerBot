
# coding: utf-8

# In[96]:

fname = r"C:\Users\Nic\Documents\pokerbot\training\smalldumb.json"
d = loadData(fname)


# In[101]:

datas = parseData(d,cardsOn=True,otherOn=False,limit = 100000)
print("Finished loading and parsing data...")


# In[111]:

print("Creating NN structure...")
# Training with 10000 games yields the following observations:
# 4 layers with 256 nDN seems to work well: 80% test acc
# 3 layers with 64 nDN seems to be the min: 70% test acc
# 100+ epochs with lr 1e-4 shows noticeable improvement, but we cap it at 100 for speed
runNN(datas[3],numEpochs=2,numDenseLayers=4,nDN=256,dropP = 0.5,lr=1e-4)


# In[106]:

# Verification that the games are well-distributed
things = datas[0]
negCount = 0
for thing in things:
    if thing[-1] == 0:
        negCount += 1
print(negCount)
print(len(things))
ratio = float(negCount)/len(things)
print(ratio)
print(len(things)*0.3)


# In[100]:

import tensorflow as tf
import json
import numpy as np
from random import shuffle
import itertools
import copy

def loadData(fname):
    datafile = open(fname, 'r')
    d = json.load(datafile)
    datafile.close()
    print("d len: " + str(len(d)))
    return d

def parseData(d,contra=False,cardsOn=True,otherOn=False,limit = -1):
    datas = [[],[],[],[]]
    # using subset of datafile for fast verification
    numSamples = len(d)#10000 if len(d) > 10000 else len(d)
    if limit > 0 and len(d) > limit:
        numSamples = limit
    snapShot = d[:numSamples]
    shuffle(snapShot)
    negCount = 0
    print(snapShot[0])
    for sample in snapShot:
        sampleDict = sample.keys()
        if sample["result"] == -1:
            negCount += 1
        hsuits=[0]*4
        for card in sample["handCards"]+sample["tableCards"]:
            if card < 13:
                hsuits[0]+=1
            elif card < 26:
                hsuits[1]+=1
            elif card < 39:
                hsuits[2]+=1
            else:
                hsuits[3]+=1
        hvalues = [0]*13
        handVals = list(map(lambda x: x % 13, sample["handCards"]+sample["tableCards"]))
        for card in handVals:
            hvalues[card] += 1
        sl = []
        if cardsOn:
            sl += hsuits+hvalues
        if 'handScore' in sampleDict:
            sl.append(float(sample["handScore"]))
            sl.append(float(sample["handRank"]))
        if 'scoreDiff' in sampleDict:
            sl.append(float(sample["scoreDiff"]))
            sl.append(float(sample["rankDiff"]))
        if otherOn:
            distFromDealer = sample["playerIndex"] - sample["dealerIndex"] % len(sample["bets"])
            remPlayers = list(map(lambda x: 1 if x else 0, sample["remPlayers"]))
            sl += sample["action"] + sample["bets"]             + [sample["pot"]] + [distFromDealer] + remPlayers
        result = 1 if sample["result"] == 1 else 0
        datas[sample["rd_num"]].append(sl+[result])
        if contra:
            datas[sample["rd_num"]].append(sl+[-sample["result"]])
    print("Negcount: " + str(negCount))
    for i in range(len(datas)):
        datas[i] = np.array(datas[i])
        print(datas[i][0])
    return datas

def weight_variable(shape):
    """ Initialize a random weight """
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    """ Initialize a random bias """
    # slight positive bias prevents dead ReLU neurons
    initial = tf.constant(0.2, shape=shape)
    return tf.Variable(initial)

def denseRelu(inFeats, weights, biases):
    return tf.nn.relu(tf.add(tf.matmul(inFeats,weights),biases))

def denseTanh(inFeats, weights, biases):
    return tf.nn.tanh(tf.add(tf.matmul(inFeats,weights),biases))

def nextBatch(currIndex,size,data):
    n = len(data)
    assert size <= n
    endIndex = currIndex+size
    if endIndex < n:
        return data[currIndex:endIndex], currIndex+size
    else:
        endIndex = endIndex-n
        return np.concatenate((data[currIndex:n],data[:endIndex]),axis=0), currIndex+size
    
def runNN(data,numEpochs=15,numDenseLayers=1,nDN=1,dropP=0.5,lr=1e-4):
    n = len(data)
    train = data[0:int(0.7*n)]
    test = data[int(0.7*n):]
    numFeatures = len(train[0])-1

    x = tf.placeholder(tf.float32, shape=[None, numFeatures])
    y_ = tf.placeholder(tf.float32, shape=[None,1])

    # Modeled after the Deep Stack architecture: multiple dense layers

    W_fc1 = weight_variable([numFeatures, nDN])
    b_fc1 = bias_variable([nDN])
    h_fc1 = denseRelu(x,W_fc1,b_fc1)

    weights = [W_fc1]
    biases = [b_fc1]
    outputs = [h_fc1]

    for i in range(numDenseLayers-1): # exclude first dense layer
        W_fc = weight_variable([nDN,nDN])
        b_fc = bias_variable([nDN])
        h_fc = denseRelu(outputs[i],W_fc,b_fc)
        weights.append(W_fc)
        biases.append(b_fc)
        outputs.append(h_fc)

    # Set keep probability for dropout layer
    keep_prob = tf.placeholder(tf.float32)
    h_fc_drop = tf.nn.dropout(outputs[-1], keep_prob)

    # Fully connected readout layer
    W_fcr = weight_variable([nDN, 1])
    b_fcr = bias_variable([nDN])
    y_result = tf.add(tf.matmul(h_fc_drop,W_fcr),b_fcr)

    loss=tf.reduce_sum(tf.square(tf.subtract(y_, y_result)))
    train_step = tf.train.AdamOptimizer(lr).minimize(loss)
    y_thresholded = tf.map_fn(lambda z: tf.cond(tf.greater(z[0],tf.constant(0.5,dtype=tf.float32)),                                                lambda:tf.constant(1,dtype=tf.float32,shape=[1]),                                                 lambda:tf.constant(0,dtype=tf.float32,shape=[1])),                              y_result, dtype = tf.float32)
    correct_prediction = tf.equal(y_thresholded, y_)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    xstart = 0
    ystart = 0
    size = max(1,int(n/100))
    numRuns = max(1,int(n/size))
    printIter = max(1,int(numEpochs/20))
    print("Starting training...")
    
    for i in range(numEpochs):
        np.random.shuffle(train)
        trainx = train[:,:-1]
        trainy = train[:,-1]
        trainy = np.reshape(trainy, (-1,1))
        for j in range(numRuns):
            xbatch,xstart = nextBatch(xstart,size,trainx)
            ybatch,ystart = nextBatch(ystart,size,trainy)
            train_step.run(feed_dict={x: xbatch, y_: ybatch, keep_prob: dropP})
        if i% printIter == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x:xbatch, y_: ybatch, keep_prob: 1.0})
            print("step %d, training accuracy %g"%(i, train_accuracy))
    
    print("Starting testing...")
    testx = test[:,:-1]
    testy = test[:,-1]
    testy = np.reshape(testy, (-1,1))
    print("test accuracy %g"%accuracy.eval(feed_dict={
        x: testx, y_: testy, keep_prob: 1.0}))
    preds = y_thresholded.eval(feed_dict={x: testx, y_:testy, keep_prob: 1.0})
    numMatch = 0
    for i in range(len(testx)):
        if preds[i] == testy[i,0]:
            numMatch+=1
    matchRatio = float(numMatch)/len(preds)
    print(matchRatio)
    sess.close()


# In[99]:

sample = {}
print(sample["blah"] == None)


# In[ ]:



