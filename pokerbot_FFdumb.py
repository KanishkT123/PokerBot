# A Feed-Forward Pokerbot implemented with TensorFlow.
# Author: Nic Trieu

import tensorflow as tf
import json
import numpy as np
from random import shuffle
import itertools
import copy
import sys

def loadData(fname):
    datafile = open(fname, 'r')
    d = json.load(datafile)
    datafile.close()
    print("d len: " + str(len(d)))
    return d

def parseData(d,contra=False,cardsOn=True,otherOn=False,limit = -1):
    data = []
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
            sl += sample["action"] + sample["bets"] + [sample["pot"]] + [distFromDealer] + remPlayers
        result = [0,1] if sample["result"] == 1 else [1,0]
        data.append(sl+result)
        if contra:
            data.append(sl+result)
    print("Negcount: " + str(negCount))
    data = np.array(data)
    if len(data) > 0:
        print(data[0])
    return data

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
    
def runNN(data,savename,numEpochs=15,numDenseLayers=1,nDN=1,dropP=0.5,lr=1e-4,doSave=False):
    n = len(data)
    train = data[0:int(0.7*n)]
    test = data[int(0.7*n):]
    numFeatures = len(test[0])-2 # ignore result one-hot

    x = tf.placeholder(tf.float32, shape=[None, numFeatures])

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
    W_fcr = weight_variable([nDN, 2])
    b_fcr = bias_variable([2])
    y_result = tf.add(tf.matmul(h_fc_drop,W_fcr),b_fcr)
    y_predict = tf.argmax(y_result,1)

    y_ = tf.placeholder(tf.float32, shape=[None,2])
    
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        labels=y_,logits=y_result))#TODO))
    train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
    correct_prediction = tf.equal(y_predict, tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
    sess.run(tf.global_variables_initializer())

    xstart = 0
    ystart = 0
    size = max(1,int(n/100))
    numRuns = max(1,int(n/size))
    printIter = 1#max(1,int(numEpochs/20))
    print("Starting training...")
    
    for i in range(numEpochs):
        np.random.shuffle(train)
        trainx = train[:,:numFeatures]
        trainy = train[:,numFeatures:]
        for j in range(numRuns):
            xbatch,xstart = nextBatch(xstart,size,trainx)
            ybatch,ystart = nextBatch(ystart,size,trainy)
            #print("trainx length: " + str(len(xbatch)))
            #print("trainy_result length: " + str(len(trainy_result)))
            #print("trainy_predict length: " + str(len(trainy_predict)))
            train_step.run(feed_dict={x: xbatch, 
                y_: ybatch, keep_prob: dropP})
        if i% printIter == 0:
            train_accuracy = sess.run(accuracy,feed_dict={x: xbatch, 
                y_: ybatch, keep_prob: 1})
            print("step %d, training accuracy %g"%(i, train_accuracy))
    
    print("Starting testing...")
    testx = test[:,:numFeatures]
    testy = test[:,numFeatures:]
    testy_predict = [0 if sample[0] > sample[1] else 1 for sample in testy]
    testacc = sess.run(accuracy,feed_dict={
        x: testx, y_:testy, keep_prob: 1.0})
    print("test accuracy %g"%testacc)
    preds = sess.run(y_predict,feed_dict={
        x: testx, y_:testy, keep_prob: 1.0})
    numMatch = 0
    for i in range(len(testx)):
        if preds[i] == testy_predict[i]:
            numMatch+=1

    for i in range(10):
        if preds[i] == testy_predict[i]:
            print("Matched! Predicted: " + str(preds[i]) + ". Actual: " + str(testy_predict[i]))
        else:
            print("Mistake... Predicted: " + str(preds[i]) + ". Actual: " + str(testy_predict[i]))
    matchRatio = float(numMatch)/len(preds)
    print(matchRatio)

    if doSave:
        saver = tf.train.Saver(write_version=tf.train.SaverDef.V1)
        save_path = saver.save(sess,savename)#.//dummy.ckpt
        print("Saved to: "+save_path)

    sess.close()

def testLoad(data,savename,numDenseLayers=1,nDN=1,dropP=0.5,lr=1e-4):

    n = len(data)
    test = data
    numFeatures = len(test[0])-2 # ignore result one-hot
    x = tf.placeholder(tf.float32, shape=[None, numFeatures])

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
    W_fcr = weight_variable([nDN, 2])
    b_fcr = bias_variable([2])
    y_result = tf.add(tf.matmul(h_fc_drop,W_fcr),b_fcr)
    y_predict = tf.argmax(y_result,1)

    #y_ = tf.placeholder(tf.float32, shape=[None,2])
    
    #cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    #    labels=y_,logits=y_result))#TODO))
    #train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
    #correct_prediction = tf.equal(y_predict, tf.argmax(y_,1))
    #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))

    saver = tf.train.Saver()
    saver.restore(sess,savename)
    print("Model restored")


    print("Starting testing...")
    testx = test[:,:numFeatures]
    testy = test[:,numFeatures:]
    testy_predict = [0 if sample[0] > sample[1] else 1 for sample in testy]
    testacc = sess.run(accuracy,feed_dict={
        x: testx, y_:testy, keep_prob: 1.0})
    print("test accuracy %g"%testacc)
    preds = sess.run(y_predict,feed_dict={
        x: testx, y_:testy, keep_prob: 1.0})
    numMatch = 0
    for i in range(len(testx)):
        if preds[i] == testy_predict[i]:
            numMatch+=1

    for i in range(10):
        if preds[i] == testy_predict[i]:
            print("Matched! Predicted: " + str(preds[i]) + ". Actual: " + str(testy_predict[i]))
        else:
            print("Mistake... Predicted: " + str(preds[i]) + ". Actual: " + str(testy_predict[i]))
    matchRatio = float(numMatch)/len(preds)
    print(matchRatio)

    sess.close()

def main():

    fname = sys.argv[1]
    limit = int(sys.argv[2])
    numEpochs = int(sys.argv[3])
    numLayers = int(sys.argv[4])
    numNeurons = int(sys.argv[5])
    dropP = float(sys.argv[6])
    lrate = float(sys.argv[7])
    doSave = bool(int(sys.argv[8]))
    trainOrLoad = int(sys.argv[9])
    savename = sys.argv[10]

    TRAIN_VAL = 1
    LOAD_VAL = 0

    if doSave:
        print("Save is ON")
    else:
        print("Save is OFF")

    #fname = "C:\\Users\\Nic\\Documents\\pokerbot\\training\\smalldumb3.json"
    d = loadData(fname)

    data = parseData(d,cardsOn=True,otherOn=False,limit = limit)
    print("Finished loading and parsing data...")

    print("Creating NN structure...")
    # Training with 10000 games yields the following observations:
    # 4 layers with 256 nDN seems to work well: 80% test acc
    # 3 layers with 64 nDN seems to be the min: 70% test acc
    # 100+ epochs with lr 1e-4 shows noticeable improvement, but we cap it at 100 for speed
    if trainOrLoad == TRAIN_VAL:
        runNN(data,savename,numEpochs=numEpochs,numDenseLayers=numLayers,nDN=numNeurons,dropP=dropP,lr=lrate,doSave=doSave)
    elif trainOrLoad == LOAD_VAL:
        testLoad(data,savename,numDenseLayers=numLayers,nDN=numNeurons,dropP=dropP,lr=lrate)

if __name__ == "__main__":
    main()