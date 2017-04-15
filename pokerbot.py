
# coding: utf-8

# In[ ]:

import tensorflow as tf
import json
import numpy as np
from random import shuffle


# In[ ]:

fname = r"C:\Users\Nic\Documents\pokerbot\training\bigdumb.json"
with open(fname, 'r') as datafile:
    d = json.load(datafile)


# In[ ]:

data = []
numSamples = 1000000 # using subset of datafile for fast verification
for sample in d[:numSamples]:
    sl = []
    for aval in sample["action"]:
        sl.append(aval)
    for bet in sample["bets"]:
        sl.append(bet)
    sl.append(sample["pot"])
    for card in sample["tableCards"]:
        sl.append(card)
    for card in sample["handCards"]:
        sl.append(card)
    distFromDealer = sample["playerIndex"] - sample["dealerIndex"] % len(sample["bets"])
    sl.append(distFromDealer)
    for player in sample["remPlayers"]:
        pval = 1 if player else 0
        sl.append(pval)
    sl.append(sample["result"])
    data.append(sl)
shuffle(data)
data = np.array(data)


# In[ ]:

n = len(data)
train = data[0:int(n*0.7)]
test = data[int(n*0.7):]


# In[ ]:

trainx = train[:,:-1]
trainy = train[:,-1]
trainy = np.reshape(trainy,(-1,1))
testx = test[:,:-1]
testy = test[:,-1]
testy = np.reshape(testy, (-1,1))
numFeatures = len(trainx[0])


# In[ ]:

def weight_variable(shape):
    """ Initialize a random weight """
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    """ Initialize a random bias """
    # slight positive bias prevents dead ReLU neurons
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def denseRelu(inFeats, weights, biases):
    return tf.nn.relu(tf.matmul(inFeats,weights))+biases

def nextBatch(currIndex,size,data):
    n = len(data)
    assert size <= n
    endIndex = currIndex+size
    if endIndex < n:
        return data[currIndex:endIndex], currIndex+size
    else:
        endIndex = endIndex-n
        return np.concatenate((data[currIndex:n],data[:endIndex]),axis=0), currIndex+size


# In[ ]:

x = tf.placeholder(tf.float32, shape=[None, numFeatures])
y_ = tf.placeholder(tf.float32, shape=[None, 1])

# Modeled after the Deep Stack architecture: 7 dense layers
numDenseLayers = 7
nDN = 500 # number of neurons per dense layer

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
b_fcr = bias_variable([1])
y_result = tf.matmul(h_fc_drop,W_fcr) + b_fcr

# Train and evaluate using ADAM optimizer
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_result))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_result,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

xstart = 0
ystart = 0
size = 1000
numRuns = int(numSamples/size)
printIter = numRuns/10
for i in range(numRuns):
    xbatch,xstart = nextBatch(xstart,size,trainx)
    ybatch,ystart = nextBatch(ystart,size,trainy)
    if i% printIter == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x:xbatch, y_: ybatch, keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
    train_step.run(feed_dict={x: xbatch, y_: ybatch, keep_prob: 0.5})

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: testx, y_: testy, keep_prob: 1.0}))

