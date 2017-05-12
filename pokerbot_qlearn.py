import pokerbot_FFdumb
import tensorflow as tf
import tableQ
import json
import numpy as np
from random import shuffle
import random
import itertools
import copy
import sys

# {'tableCards': [], 
#'rd_num': 0, 
#'pot': 150, 
#'action': (1, 50), 
#'playerIndex': 1, 
#'dealerIndex': 1, 
#'remPlayers': [True, True], 
#'result': -1, 
#'handCards': [38, 49], 
#'callBet': 50, 
#'bets': [100, 100]}

def loadData(fname):
    datafile = open(fname, 'r')
    d = json.load(datafile)
    datafile.close()
    print("d len: " + str(len(d)))
    return d

def parseData(d,cardsOn=True,otherOn=True,limit = -1):#TODO: match limit with tables
    data = []
    # using subset of datafile for fast verification
    numSamples = len(d)#10000 if len(d) > 10000 else len(d)
    print("numSamples: " + str(numSamples))
    if limit > 0 and len(d) > limit:
        numSamples = limit
    snapShot = d[:numSamples]
    shuffle(snapShot)
    print(snapShot[0])
    playerIndices = []
    for sample in snapShot:
        playerIndices.append(sample["playerIndex"])
        sampleDict = sample.keys()
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
            sl += sample["bets"] + [sample["pot"]] + [distFromDealer] + remPlayers
        data.append(sl)
    #data = np.array(data)
    if len(data) > 0:
        print(data[0])
    return data,playerIndices

def parseFeatDict(featDict,cardsOn=True,otherOn=False):
    sample = featDict
    sampleDict = sample.keys()
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
        sl += sample["bets"] + [sample["pot"]] + [distFromDealer] + remPlayers
    return sl

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
    
### JUMPPOINT
def runNN(featVecs,numCurrDumbFeats,playerIndices,tableDicts,savenameCurr,savenameDumbNext=None,savenameQNext=None,
    savenameQ=None,randBase=0.1,numFeatures=21,numFeaturesNext=None,
    numDenseLayers=4,nDN=128,dropP=0.5,lr=1e-4, qRatio = 0.2, doSave = False):
    # -------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------
    # Load FFdumb net for the current round
    print("Loading FFdumb net for current round...")
    gDumbCurr = tf.Graph()
    with gDumbCurr.as_default():
        x = tf.placeholder(tf.float32, shape=[None, numCurrDumbFeats])
        W_fc1 = weight_variable([numCurrDumbFeats, nDN])
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
        keep_prob = tf.placeholder(tf.float32)
        h_fc_drop = tf.nn.dropout(outputs[-1], keep_prob)
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
        
        saverDumbCurr = tf.train.Saver()
        varDictDumbCurr = {"y_predict" : y_predict, "x" : x, "y_" : y_, "keep_prob" : keep_prob}

    sessDumbCurr = tf.InteractiveSession(graph = gDumbCurr)
    saverDumbCurr.restore(sessDumbCurr,savenameCurr)
    # -------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------
    # Load FFdumb net for the next round
    if savenameDumbNext != None:
        print("Loading FFdumb net for next round...")
        gDumbNext = tf.Graph()
        with gDumbNext.as_default():
            x = tf.placeholder(tf.float32, shape=[None, numFeaturesNext])
            W_fc1 = weight_variable([numFeaturesNext, nDN])
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
            keep_prob = tf.placeholder(tf.float32)
            h_fc_drop = tf.nn.dropout(outputs[-1], keep_prob)
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
            
            saverDumbNext = tf.train.Saver()
            varDictDumbNext = {"y_predict" : y_predict, "x" : x, "y_" : y_, "keep_prob" : keep_prob}

        sessDumbNext = tf.InteractiveSession(graph = gDumbNext)
        saverDumbNext.restore(sessDumbNext,savenameDumbNext)
    # -------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------
    # Load FFQ net for next round
    if savenameQNext != None:
        print("Loading FFQ net for next round...")
        gQNext = tf.Graph()
        with gQNext.as_default():
            x = tf.placeholder(tf.float32, shape=[None,numFeaturesNext])
            W_fc1 = weight_variable([numFeaturesNext, nDN])
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
            keep_prob = tf.placeholder(tf.float32)
            h_fc_drop = tf.nn.dropout(outputs[-1], keep_prob)
            W_fcr = weight_variable([nDN, 4])
            b_fcr = bias_variable([4])
            y_result = tf.add(tf.matmul(h_fc_drop,W_fcr),b_fcr)
            y_predict = tf.argmax(y_result,1)
            y_ = tf.placeholder(tf.float32, shape=[None,4])
            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                labels=y_,logits=y_result))#TODO))
            train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
            correct_prediction = tf.equal(y_predict, tf.argmax(y_,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            saverQNext = tf.train.Saver()
            varDictQNext = {"y_predict" : y_predict, "y_result" : y_result, "x" : x, "y_" : y_, "keep_prob" : keep_prob}
        sessQNext = tf.InteractiveSession(graph = gQNext)
        saverQNext.restore(sessQNext,savenameQNext)
    
    # -------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------
    print("Constructing FFQ net for current round...")
    # Construct FFQ net
    gQ = tf.Graph()
    with gQ.as_default():
        x = tf.placeholder(tf.float32, shape=[None,numFeatures])
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
        keep_prob = tf.placeholder(tf.float32)
        h_fc_drop = tf.nn.dropout(outputs[-1], keep_prob)
        W_fcr = weight_variable([nDN, 3])
        b_fcr = bias_variable([3])
        y_result = tf.add(tf.matmul(h_fc_drop,W_fcr),b_fcr)
        y_predict = tf.argmax(y_result,1)
        y_ = tf.placeholder(tf.float32, shape=[None,3])
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=y_,logits=y_result))#TODO))
        train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
        correct_prediction = tf.equal(y_predict, tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        saverQ = tf.train.Saver()
        varDictQ = {"y_predict" : y_predict, "y_result" : y_result, "x" : x, "y_" : y_, "keep_prob" : keep_prob, "train_step" : train_step}
    sessQ = tf.InteractiveSession(graph = gQ)
    sessQ.run(tf.global_variables_initializer())
    # -------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------
    # Run the Q-learning
    print("Training FFQ net via Q-learning...")
    players = [tableQ.Player(),tableQ.Player()]
    t = tableQ.Table(players)    
    for i in range(len(tableDicts)):
        tableDict = tableDicts[i]
        playerIndex = playerIndices[i]
        featVec = np.array([featVecs[i]])
        resultFiller = np.array([[-1]*3])
        action,allQ = sessQ.run([varDictQ["y_predict"],varDictQ["y_result"]],
            feed_dict={varDictQ["x"]:featVec,varDictQ["y_"]: resultFiller, varDictQ["keep_prob"]: 1})
        allQ = list(allQ[0])
        action = action[0]
        
        # Sometimes, choose action randomly to explore
        if np.random.rand(1) < randBase:
            action = random.randint(0,2)
        bets = tableDict["bets"]
        maxBet = 0
        for bet in bets:
            if bet > maxBet:
                maxBet = bet
        currBet = bets[playerIndex]
        callBet = maxBet-currBet
        potBet = callBet+tableDict["pot"]
        bet = 0
        if action == tableQ.BET_VAL or action == tableQ.ALLIN_VAL:
            if tableDict["allIn"]:
                action = tableQ.CALL_VAL
        if action == tableQ.CALL_VAL:
            bet = callBet
        elif action == tableQ.BET_VAL:
            bet = potBet
        elif action == tableQ.ALLIN_VAL:
            bet = tableQ.STACK - currBet

        if bet + currBet > tableQ.STACK:
            bet = tableQ.STACK - currBet
            action = tableQ.ALLIN_VAL

        if action == tableQ.FOLD_VAL:
            # Terminal state: If we fold, lose the current bet
            r = -currBet
            maxNextQ = 0
        else:
            # Get new state and reward from environment
            tableDict = tableDicts[i]
            allPlayerCards = tableDict["allPlayerCards"]

            # Set all players to dumb for the simulation;
            # The Q-learning player won't actually be playing since we train on 1 rd.
            nnNext = None
            varDictNext = None
            if savenameQNext != None:
                nnNext = sessQNext
                varDictNext = varDictQNext
            elif savenameDumbNext != None:
                nnNext = sessDumbNext
                varDictNext = varDictDumbNext
            playerDumb1 = tableQ.Player(hand=allPlayerCards[0],nn=sessDumbCurr,nnNext=nnNext,dumb=True,varDict = varDictDumbCurr,varDictNext=varDictNext)
            playerDumb2 = tableQ.Player(hand=allPlayerCards[1],nn=sessDumbCurr,nnNext=nnNext,dumb=True,varDict = varDictDumbCurr,varDictNext=varDictNext)
            players = [playerDumb1,playerDumb2]

            # For the final round, we get a list of winners.
            # For the other rounds, we get a feature dict of the next state
            currRd_num = tableDict["rd_num"]
            assert (tableDict["pot"] != 0)
            gameDone,featDict,winnerList = t.simulateStep(tableDict,players,playerIndex,action,bet)
            assert (featDict["pot"] != 0)
            if gameDone: # Terminal state: win pot/lose bet
                maxNextQ = 0
                if playerIndex in winnerList:
                    r = featDict["pot"]/len(winnerList)
                else:
                    r = -(currBet + bet)
            else:
                # If game not finished, leave reward at 0 and calculate next max Qs
                r = 0
                # If the next state is in round 3, use FFDumb for the next round
                # Else use the QLearning network
                nextRd_num = featDict["rd_num"]
                nextPot = featDict["pot"]
                if nextRd_num == 3:
                    # Use FFDumb for the next round
                    parsedFeats = np.array([parseFeatDict(featDict,cardsOn=True,otherOn=False)])
                    resultFiller = np.array([[-1]*2])
                    predictedResult = sessDumbNext.run(varDictDumbNext["y_predict"],
                        feed_dict={varDictDumbNext["x"]:parsedFeats,varDictDumbNext["y_"]:resultFiller,varDictDumbNext["keep_prob"]:1})
                    if predictedResult == 1:
                        # Predicted win
                        maxNextQ = nextPot
                    else:
                        # Predicted loss
                        maxNextQ = -(currBet+bet)

                elif currRd_num == nextRd_num:
                    # TODO: Use current static QLearning network
                    # Right now, using current dumb network as approximator
                    parsedFeats = np.array([parseFeatDict(featDict,cardsOn=True,otherOn=False)])
                    resultFiller = np.array([[-1]*2])
                    predictedResult = sessDumbCurr.run(varDictDumbCurr["y_predict"],
                        feed_dict={varDictDumbCurr["x"]:parsedFeats,varDictDumbCurr["y_"]: resultFiller, varDictDumbCurr["keep_prob"]: 1})
                    if predictedResult == 1:
                        # Predicted win
                        maxNextQ = nextPot
                    else:
                        # Predicted loss
                        maxNextQ = -(currBet+bet)
                else:
                    parsedFeats = np.array([parseFeatDict(featDict,cardsOn=True,otherOn=True)])
                    resultFiller = np.array([[-1]*3])
                    nextQs = sessQNext.run(varDictQNext["y_result"], 
                        feed_dict={varDictQNext["x"]:featVec,varDictQNext["y_"]: resultFiller, varDictQNext["keep_prob"]: 1})
                    nextQs = list(nextQs[0])
                    maxNextQ = np.max(nextQs)
        
        allQ[action] = allQ[action] + r + qRatio*maxNextQ
        allQ = np.array([allQ])
        
        #Train our network using target and predicted Q values
        sessQ.run(varDictQ["train_step"],feed_dict={x:featVec,y_:allQ, keep_prob: dropP})
        if i % 100 == 0:
            print("Iteration " + str(i) + ". Reward found: " + str(round(r,2)) + ". MaxQ found: " + str(round(maxNextQ,2)))

    if doSave:
        savepath = saverQ.save(sessQ,savenameQ)
        print("Saved to: "+savepath)

def main():

    fname = sys.argv[1]
    randBase = float(sys.argv[2])
    qRatio = float(sys.argv[3])
    numLayers = int(sys.argv[4])
    numNeurons = int(sys.argv[5])
    dropP = float(sys.argv[6])
    lrate = float(sys.argv[7])
    doSave = bool(int(sys.argv[8])) # 1 for save, 0 for no save
    dumbNextorQNext = int(sys.argv[9]) # 1 for dumb, 0 for q, -1 for none
    savenameCurr = sys.argv[10]
    savenameNext = sys.argv[11]
    savenameQ = sys.argv[12]
    fnameCurrMicro = sys.argv[13] # micro of next round
    fnameNextMicro = sys.argv[14] # micro of this round


    DUMB_VAL = 1
    Q_VAL = 0

    if doSave:
        print("Save is ON")
    else:
        print("Save is OFF")

    #fname = "C:\\Users\\Nic\\Documents\\pokerbot\\training\\smalldumb3.json"
    d = loadData(fname)
    featDicts = [thing[0] for thing in d]
    tableDicts = [thing[1] for thing in d]
    featVecs,playerIndices = parseData(featDicts,cardsOn=True,otherOn=True,limit = -1)
    numCurrQFeats = len(featVecs[0])   

    dDumb = loadData(fnameCurrMicro)
    featDictsDumb = [thing[0] for thing in d]
    featVecsDumb,_ = parseData(featDictsDumb,cardsOn=True,otherOn=False,limit=-1)
    numCurrDumbFeats = len(featVecsDumb[0])

    print("Finished loading and parsing data...")

    print("Creating NN structure...")
    # Training with 10000 games yields the following observations:
    # 4 layers with 256 nDN seems to work well: 80% test acc
    # 3 layers with 64 nDN seems to be the min: 70% test acc
    # 100+ epochs with lr 1e-4 shows noticeable improvement, but we cap it at 100 for speed
    if dumbNextorQNext == Q_VAL:
        dNext = loadData(fnameNextMicro)
        featDictsNext = [thing[0] for thing in dNext]
        featVecsNext,_ = parseData(featDictsNext, cardsOn = True, otherOn=True,limit=-1)
        numNextFeats = len(featVecsNext)
        runNN(featVecs,numCurrDumbFeats,playerIndices,tableDicts,savenameCurr,savenameDumbNext=None,savenameQNext=savenameNext,
            savenameQ=savenameQ,randBase=randBase,numFeatures=numCurrQFeats,numFeaturesNext=numNextFeats,
            numDenseLayers=numLayers,nDN=numNeurons,dropP=dropP,lr=lrate, qRatio = 0.2, doSave = doSave)
    elif dumbNextorQNext == DUMB_VAL:
        dNext = loadData(fnameNextMicro)
        featDictsNext = [thing[0] for thing in dNext]
        featVecsNext,_ = parseData(featDictsNext, cardsOn = True, otherOn=False,limit=-1)
        numNextFeats = len(featVecsNext[0])
        runNN(featVecs,numCurrDumbFeats,playerIndices,tableDicts,savenameCurr,savenameDumbNext=savenameNext,savenameQNext=None,
            savenameQ=savenameQ,randBase=randBase,numFeatures=numCurrQFeats,numFeaturesNext=numNextFeats,
            numDenseLayers=numLayers,nDN=numNeurons,dropP=dropP,lr=lrate, qRatio = qRatio, doSave = doSave)
    else:
        runNN(featVecs,numCurrDumbFeats,playerIndices,tableDicts,savenameCurr,savenameDumbNext=None,savenameQNext=None,
            savenameQ=savenameQ,randBase=randBase,numFeatures=numCurrQFeats,numFeaturesNext=None,
            numDenseLayers=numLayers,nDN=numNeurons,dropP=dropP,lr=lrate, qRatio = qRatio, doSave = doSave)
    
if __name__ == "__main__":
    main()