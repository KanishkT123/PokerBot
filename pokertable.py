import random

SMALL_BLIND = 50
BIG_BLIND = 100
STACK = 200*BIG_BLIND

class Player():
    def __init__(self,stack):
        self.hand = []
    
    def act(self,dealerDist,tableCards,pot,bets,remainingPlayers,rd_num):
        """
        Reponse Values:
        0: Fold
        1: Call
        2: Raise
        
        Returns reponse, bet
        """
        #TODO: implement
        return 0,0
        
class Table():
    def __init__(self,players):
        self.numPlayers = len(players)
        self.players = players
        self.numCardsToDeal = self.numPlayers*2+3
        assert(self.numCardsToDeal <= 52)
        
        self.cardsToDeal = [0]*numCards
        self.dealIndex = 0
        self.pot = 0
        self.bets = [0]*self.numPlayers
        self.remainingPlayers = [True]*self.numPlayers
        self.checks = [False]*self.numPlayers
        self.tableCards = [None]*5
        
    def resetTable(self):
        self.dealIndex = 0
        self.pot = 0
        self.bets = [0]*self.numPlayers
        self.remainingPlayers = [True]*self.numPlayers
        self.tableCards = [None]*5
        
        viableCards = list(range(52))
        for i in range(self.numCardsToDeal):
            cardIndex = random.randint(0,len(viableCards)-1)
            self.cardsToDeal[i] = viableCards[cardIndex]
            del(viableCards[cardIndex])
        
    def playGame(self,dealerIndex):
        self.resetTable()
        
        self.preFlop(dealerIndex)
        self.flop()
        self.turn()
        winplayer = self.river()
    
    def computeBestHand(self,player):
        # TODO: implement with pokerstove: https://github.com/andrewprock/pokerstove
        return -1
    
    def doBets(self,rd_num,dealerIndex):
        firstBetPos = dealerIndex if rd_num == 0 else dealerIndex+1
        while not allChecked():
            for i in range(self.numPlayers):
                i = (i+firstBetPos) % self.numPlayers
                if not self.remainingPlayers[i]:
                    continue
                response,bet = self.players[i].act(
                    i,self.tableCards,self.pot,self.bets,self.remainingPlayers,rd_num)
                if response == 0:
                    self.checks[i] = True
                    self.remainingPlayers[i] = False
                elif response == 1:
                    self.checks[i] = True
                    self.bets[i] += bet
                    self.pot += bet
                elif response == 2:
                    self.checks[i] = False
                    self.bets[i] += bet
                    self.pot += bet
    
    def preFlop(self,dealerIndex):
        for player in self.players:
            card1 = self.cardsToDeal[self.dealIndex]
            card2 = self.cardsToDeal[self.dealIndex+1]
            player.hand = [card1,card2]
            self.dealIndex += 2
        self.bets[dealerIndex] = SMALL_BLIND
        self.bets[(dealerIndex+1) % self.numPlayers] = BIG_BLIND
        self.pot = SMALL_BLIND + BIG_BLIND
        self.doBets(0,dealerIndex)
    
    def flop(self,dealerIndex):
        for i in [0,1,2]:
            self.tableCards[i] = self.cardsToDeal[self.dealIndex]
            self.dealIndex += 1
        self.doBets(1,dealerIndex)
    
    def turn(self,dealerIndex):
        self.tableCards[3] = self.cardsToDeal[self.dealIndex]
        self.dealIndex += 1
        self.doBets(2,dealerIndex)
        
    def river(self):
        self.tableCards[4] = self.cardsToDeal[self.dealIndex]
        self.dealIndex += 1
        self.doBets(3,dealerIndex)
        
        winhand = -1
        winplayer = -1
        for i in range(self.numPlayers):
            if not self.remainingPlayers[i]:
                continue
            player = self.players[i]
            hand = self.computeBestHand(player)
            if winhand < hand:
                winhand = hand
                winplayer = i
        return winplayer

