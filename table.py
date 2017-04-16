
# coding: utf-8

# In[84]:

import random
import json
from random import shuffle
import itertools

SMALL_BLIND = 50
BIG_BLIND = 100
STACK = 200*BIG_BLIND
DUMB_THRESHOLD_SMALL = 500
DUMB_THRESHOLD_BIG = 1500
FOLD_VAL = 0
CALL_VAL = 1
BET_VAL = 2

def generateTrainingData(numSamples, fname):
    """
    Just call this to simulate poker games and save to a json file.
    - numSamples: positive integer number of games to simulate
    - fname: a string denoting a json file location
    """
    players = [Player(),Player()]
    t = Table(players)    
    with open(fname,'w') as outfile:
        actDicts = []
        for i in range(numSamples):
            # TODO: might want to randomize the dealer index
            actDicts += t.playGame(0,log=True)
            if i % 10000 == 0:
                print(i)
        print("dumping")
        json.dump(actDicts,outfile)
        outfile.close()
        
def permutation(current, indices):
    """
    Permutes a certain section of a list and returns all permutations
        -current: current list/input list
        -indices: All indices to be permuted (assumes that all indices
                  sequential)
    """
    indices.sort()
	permuter = [current[a] for a in indices]
	permutations = [list(x) for x in list(itertools.permutations(permuter))]
	temp1 = current[:min(indices)]
	temp2 = current[max(indices)+1:]
	alllist = []
	for i in permutations:
		alllist.append(temp1 + i + temp2)
	return alllist


class Player():
    def __init__(self):
        self.hand = []
    
    def dumbAct(self,dealerIndex,playerIndex,tableCards,pot,
                bets,remainingPlayers,rd_num,evaluator,log=False,onlyCall=False):
        """
        Reponses:
            0: Fold
            1: Call
            2: Bet pot
        
        Saves the decision and features used in actDict.        
        Returns (Response, bet amount, actDict).
        """
        maxBet = 0
        for bet in bets:
            if bet > maxBet:
                maxBet = bet
        currBet = bets[playerIndex]
        callBet = maxBet-currBet
        potBet = callBet+pot
        descriptorString = None
        actDict = {}
        if log:
            actDict["dealerIndex"] = dealerIndex
            actDict["playerIndex"] = playerIndex
            actDict["handCards"] = self.hand
            actDict["tableCards"] = tableCards
            actDict["pot"] = pot
            actDict["bets"] = bets
            actDict["remPlayers"] = remainingPlayers
            actDict["rd_num"] = rd_num
        if len(tableCards) < 5:
            action = (CALL_VAL,callBet)
            if log:
                actDict["action"] = action
            return (CALL_VAL,callBet,actDict)
        playerScore = evaluator.evaluate(self.hand,tableCards)
        tableScore = evaluator.evaluate([],tableCards)
        if potBet > STACK:
            potBet = STACK
        if (playerScore > max(tableScore - DUMB_THRESHOLD_SMALL,1) and tableScore < 3325) or playerScore > max(tableScore - DUMB_THRESHOLD_BIG,1) or onlyCall:
            action = (CALL_VAL,callBet)
            if log:
                actDict["action"] = action
            return (CALL_VAL,callBet,actDict)
        else:
            action = (BET_VAL,potBet)
            if log:
                actDict["action"] = action
            return (BET_VAL,potBet,actDict)
        
class Table():
    def __init__(self,players):
        self.numPlayers = len(players)
        self.players = players
        self.numCardsToDeal = self.numPlayers*2+5
        assert(self.numCardsToDeal <= 52)
        
        self.cardsToDeal = [0]*self.numCardsToDeal
        self.dealIndex = 0
        self.pot = 0
        self.bets = [0]*self.numPlayers
        self.remainingPlayers = [True]*self.numPlayers
        self.checks = [False]*self.numPlayers
        self.tableCards = []
        self.evaluator = Evaluator()
        self.logs = []
        
    def resetTable(self):
        self.dealIndex = 0
        self.pot = 0
        self.bets = [0]*self.numPlayers
        self.remainingPlayers = [True]*self.numPlayers
        self.tableCards = []
        self.checks = [False]*self.numPlayers
        self.logs = []
        
        viableCards = Deck.GetFullDeck()
        for i in range(self.numCardsToDeal):
            cardIndex = random.randint(0,len(viableCards)-1)
            self.cardsToDeal[i] = viableCards[cardIndex]
            del(viableCards[cardIndex])
        
    def playGame(self,dealerIndex,log):
        self.resetTable()
        self.preFlop(dealerIndex,log)
        self.flop(dealerIndex,log)
        self.turn(dealerIndex,log)
        winplayers = self.river(dealerIndex,log)
        if log:
            for actDict in self.logs:
                playerIndex = actDict["playerIndex"]
                actDict["result"] = 1 if playerIndex in winplayers else -1
        return self.logs
    
    def computeBestHand(self,player):
        return self.evaluator.evaluate(player.hand,self.tableCards)
    
    def allChecked(self):
        for check in self.checks:
            if check == False:
                return False
        return True
    
    def doBets(self,rd_num,dealerIndex, log):
        firstBetPos = dealerIndex if rd_num == 0 else dealerIndex+1
        self.checks = [False]*self.numPlayers
        for betround in range(2):
            for i in range(self.numPlayers):
                if self.allChecked():
                    break
                i = (i+firstBetPos) % self.numPlayers
                if not self.remainingPlayers[i]:
                    self.checks[i] = True
                    continue
                response,bet,actDict = self.players[i].dumbAct(
                    dealerIndex,i,self.tableCards,self.pot,
                    self.bets,self.remainingPlayers,rd_num,self.evaluator,
                    log = log, onlyCall = betround==1)
                if log:
                    self.logs.append(actDict)
                if response == FOLD_VAL:
                    self.checks[i] = True
                    self.remainingPlayers[i] = False
                elif response == CALL_VAL:
                    self.checks[i] = True
                    self.bets[i] += bet
                    self.pot += bet
                elif response == BET_VAL:
                    self.checks[i] = False
                    self.bets[i] += bet
                    self.pot += bet
    
    def preFlop(self,dealerIndex,log):
        for player in self.players:
            card1 = self.cardsToDeal[self.dealIndex]
            card2 = self.cardsToDeal[self.dealIndex+1]
            player.hand = [card1,card2]
            self.dealIndex += 2
        self.bets[dealerIndex] = SMALL_BLIND
        self.bets[(dealerIndex+1) % self.numPlayers] = BIG_BLIND
        self.pot = SMALL_BLIND + BIG_BLIND
        self.doBets(0,dealerIndex,log)
    
    def flop(self,dealerIndex,log):
        for i in range(3):
            self.tableCards.append(self.cardsToDeal[self.dealIndex])
            self.dealIndex += 1
        self.doBets(1,dealerIndex,log)
    
    def turn(self,dealerIndex,log):
        self.tableCards.append(self.cardsToDeal[self.dealIndex])
        self.dealIndex += 1
        self.doBets(2,dealerIndex,log)
        
    def river(self,dealerIndex,log):
        self.tableCards.append(self.cardsToDeal[self.dealIndex])
        self.dealIndex += 1
        self.doBets(3,dealerIndex,log)
        
        winhand = float('inf')
        winplayers = []
        for i in range(self.numPlayers):
            if not self.remainingPlayers[i]:
                continue
            player = self.players[i]
            hand = self.computeBestHand(player)
            #print(hand)
            if winhand > hand:
                winhand = hand
                winplayers = [i]
            elif winhand == hand:
                winplayers.append(i)
        return winplayers

# END OF OUR CODE
    


# In[4]:


# ALEX BELOI'S CODE STARTS HERE
# https://github.com/alexbeloi/nn-holdem

class Card ():
    """
    Static class that handles cards. We represent cards as 32-bit integers, so
    there is no object instantiation - they are just ints. Most of the bits are
    used, and have a specific meaning. See below:

                                    Card:

                          bitrank     suit rank   prime
                    +--------+--------+--------+--------+
                    |xxxbbbbb|bbbbbbbb|cdhsrrrr|xxpppppp|
                    +--------+--------+--------+--------+

        1) p = prime number of rank (deuce=2,trey=3,four=5,...,ace=41)
        2) r = rank of card (deuce=0,trey=1,four=2,five=3,...,ace=12)
        3) cdhs = suit of card (bit turned on based on suit of card)
        4) b = bit turned on depending on rank of card
        5) x = unused

    This representation will allow us to do very important things like:
    - Make a unique prime prodcut for each hand
    - Detect flushes
    - Detect straights

    and is also quite performant.
    """

    # the basics
    STR_RANKS = '23456789TJQKA'
    INT_RANKS = list(range(13))
    PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41]

    # converstion from string => int
    CHAR_RANK_TO_INT_RANK = dict(list(zip(list(STR_RANKS), INT_RANKS)))
    CHAR_SUIT_TO_INT_SUIT = {
        's' : 1, # spades
        'h' : 2, # hearts
        'd' : 4, # diamonds
        'c' : 8, # clubs
    }
    INT_SUIT_TO_CHAR_SUIT = 'xshxdxxxc'

    # for pretty printing
    PRETTY_SUITS = {
        1 : "\u2660".encode('utf-8').decode(), # spades
        2 : "\u2764".encode('utf-8').decode(), # hearts
        4 : "\u2666".encode('utf-8').decode(), # diamonds
        8 : "\u2663".encode('utf-8').decode() # clubs
    }

     # hearts and diamonds
    PRETTY_REDS = [2, 4]

    @staticmethod
    def new(string):
        """
        Converts Card string to binary integer representation of card, inspired by:

        http://www.suffecool.net/poker/evaluator.html
        """

        rank_char = string[0]
        suit_char = string[1]
        rank_int = Card.CHAR_RANK_TO_INT_RANK[rank_char]
        suit_int = Card.CHAR_SUIT_TO_INT_SUIT[suit_char]
        rank_prime = Card.PRIMES[rank_int]

        bitrank = 1 << rank_int << 16
        suit = suit_int << 12
        rank = rank_int << 8

        return bitrank | suit | rank | rank_prime

    @staticmethod
    def int_to_str(card_int):
        rank_int = Card.get_rank_int(card_int)
        suit_int = Card.get_suit_int(card_int)
        return Card.STR_RANKS[rank_int] + Card.INT_SUIT_TO_CHAR_SUIT[suit_int]

    @staticmethod
    def get_rank_int(card_int):
        return (card_int >> 8) & 0xF

    @staticmethod
    def get_suit_int(card_int):
        return (card_int >> 12) & 0xF

    @staticmethod
    def get_bitrank_int(card_int):
        return (card_int >> 16) & 0x1FFF

    @staticmethod
    def get_prime(card_int):
        return card_int & 0x3F

    @staticmethod
    def hand_to_binary(card_strs):
        """
        Expects a list of cards as strings and returns a list
        of integers of same length corresponding to those strings.
        """
        bhand = []
        for c in card_strs:
            bhand.append(Card.new(c))
        return bhand

    @staticmethod
    def prime_product_from_hand(card_ints):
        """
        Expects a list of cards in integer form.
        """

        product = 1
        for c in card_ints:
            product *= (c & 0xFF)

        return product

    @staticmethod
    def prime_product_from_rankbits(rankbits):
        """
        Returns the prime product using the bitrank (b)
        bits of the hand. Each 1 in the sequence is converted
        to the correct prime and multiplied in.

        Params:
            rankbits = a single 32-bit (only 13-bits set) integer representing
                    the ranks of 5 _different_ ranked cards
                    (5 of 13 bits are set)

        Primarily used for evaulating flushes and straights,
        two occasions where we know the ranks are *ALL* different.

        Assumes that the input is in form (set bits):

                              rankbits
                        +--------+--------+
                        |xxxbbbbb|bbbbbbbb|
                        +--------+--------+

        """
        product = 1
        for i in Card.INT_RANKS:
            # if the ith bit is set
            if rankbits & (1 << i):
                product *= Card.PRIMES[i]

        return product

    @staticmethod
    def int_to_binary(card_int):
        """
        For debugging purposes. Displays the binary number as a
        human readable string in groups of four digits.
        """
        bstr = bin(card_int)[2:][::-1] # chop off the 0b and THEN reverse string
        output = list("".join(["0000" +"\t"] * 7) +"0000")

        for i in range(len(bstr)):
            output[i + int(i/4)] = bstr[i]

        # output the string to console
        output.reverse()
        return "".join(output)

    @staticmethod
    def int_to_pretty_str(card_int):
        """
        Prints a single card
        """

        color = False
        try:
            from termcolor import colored
            ### for mac, linux: http://pypi.python.org/pypi/termcolor
            ### can use for windows: http://pypi.python.org/pypi/colorama
            color = True
        except ImportError:
            pass

        # suit and rank
        suit_int = Card.get_suit_int(card_int)
        rank_int = Card.get_rank_int(card_int)

        # if we need to color red
        s = Card.PRETTY_SUITS[suit_int]
        if color and suit_int in Card.PRETTY_REDS:
            s = colored(s, "red")

        r = Card.STR_RANKS[rank_int]

        return " [ " +r+ " " +s+ " ] "

    @staticmethod
    def print_pretty_card(card_int):
        """
        Expects a single integer as input
        """
        print(Card.int_to_pretty_str(card_int))

    @staticmethod
    def print_pretty_cards(card_ints):
        """
        Expects a list of cards in integer form.
        """
        output = " "
        for i in range(len(card_ints)):
            c = card_ints[i]
            if i != len(card_ints) - 1:
                output += Card.int_to_pretty_str(c) + ","
            else:
                output += Card.int_to_pretty_str(c) + " "

        print(output)
        
class Deck:
    """
    Class representing a deck. The first time we create, we seed the static
    deck with the list of unique card integers. Each object instantiated simply
    makes a copy of this object and shuffles it.
    """
    _FULL_DECK = []

    def __init__(self):
        self.shuffle()

    def shuffle(self):
        # and then shuffle
        self.cards = Deck.GetFullDeck()
        #shuffle(self.cards)

    def draw(self, n=1):
        if n == 1:
            return self.cards.pop(0)

        cards = []
        for i in range(n):
            cards.append(self.draw())
        return cards

    def __str__(self):
        return Card.print_pretty_cards(self.cards)

    @staticmethod
    def GetFullDeck():
        if Deck._FULL_DECK:
            return list(Deck._FULL_DECK)

        # create the standard 52 card deck
        for rank in Card.STR_RANKS:
            for suit,val in Card.CHAR_SUIT_TO_INT_SUIT.items():
                Deck._FULL_DECK.append(Card.new(rank + suit))

        return list(Deck._FULL_DECK)

class LookupTable(object):
    """
    Number of Distinct Hand Values:

    Straight Flush   10
    Four of a Kind   156      [(13 choose 2) * (2 choose 1)]
    Full Houses      156      [(13 choose 2) * (2 choose 1)]
    Flush            1277     [(13 choose 5) - 10 straight flushes]
    Straight         10
    Three of a Kind  858      [(13 choose 3) * (3 choose 1)]
    Two Pair         858      [(13 choose 3) * (3 choose 2)]
    One Pair         2860     [(13 choose 4) * (4 choose 1)]
    High Card      + 1277     [(13 choose 5) - 10 straights]
    -------------------------
    TOTAL            7462

    Here we create a lookup table which maps:
        5 card hand's unique prime product => rank in range [1, 7462]

    Examples:
    * Royal flush (best hand possible)          => 1
    * 7-5-4-3-2 unsuited (worst hand possible)  => 7462
    """
    MAX_STRAIGHT_FLUSH  = 10
    MAX_FOUR_OF_A_KIND  = 166
    MAX_FULL_HOUSE      = 322
    MAX_FLUSH           = 1599
    MAX_STRAIGHT        = 1609
    MAX_THREE_OF_A_KIND = 2467
    MAX_TWO_PAIR        = 3325
    MAX_PAIR            = 6185
    MAX_HIGH_CARD       = 7462

    MAX_TO_RANK_CLASS = {
        MAX_STRAIGHT_FLUSH: 1,
        MAX_FOUR_OF_A_KIND: 2,
        MAX_FULL_HOUSE: 3,
        MAX_FLUSH: 4,
        MAX_STRAIGHT: 5,
        MAX_THREE_OF_A_KIND: 6,
        MAX_TWO_PAIR: 7,
        MAX_PAIR: 8,
        MAX_HIGH_CARD: 9
    }

    RANK_CLASS_TO_STRING = {
        1 : "Straight Flush",
        2 : "Four of a Kind",
        3 : "Full House",
        4 : "Flush",
        5 : "Straight",
        6 : "Three of a Kind",
        7 : "Two Pair",
        8 : "Pair",
        9 : "High Card"
    }

    def __init__(self):
        """
        Calculates lookup tables
        """
        # create dictionaries
        self.flush_lookup = {}
        self.unsuited_lookup = {}

        # create the lookup table in piecewise fashion
        self.flushes()  # this will call straights and high cards method,
                        # we reuse some of the bit sequences
        self.multiples()

    def flushes(self):
        """
        Straight flushes and flushes.

        Lookup is done on 13 bit integer (2^13 > 7462):
        xxxbbbbb bbbbbbbb => integer hand index
        """

        # straight flushes in rank order
        straight_flushes = [
            7936, # int('0b1111100000000', 2), # royal flush
            3968, # int('0b111110000000', 2),
            1984, # int('0b11111000000', 2),
            992, # int('0b1111100000', 2),
            496, # int('0b111110000', 2),
            248, # int('0b11111000', 2),
            124, # int('0b1111100', 2),
            62, # int('0b111110', 2),
            31, # int('0b11111', 2),
            4111 # int('0b1000000001111', 2) # 5 high
        ]

        # now we'll dynamically generate all the other
        # flushes (including straight flushes)
        flushes = []
        gen = self.get_lexographically_next_bit_sequence(int('0b11111', 2))

        # 1277 = number of high cards
        # 1277 + len(str_flushes) is number of hands with all cards unique rank
        for i in range(1277 + len(straight_flushes) - 1): # we also iterate over SFs
            # pull the next flush pattern from our generator
            f = next(gen)

            # if this flush matches perfectly any
            # straight flush, do not add it
            notSF = True
            for sf in straight_flushes:
                # if f XOR sf == 0, then bit pattern
                # is same, and we should not add
                if not f ^ sf:
                    notSF = False

            if notSF:
                flushes.append(f)

        # we started from the lowest straight pattern, now we want to start ranking from
        # the most powerful hands, so we reverse
        flushes.reverse()

        # now add to the lookup map:
        # start with straight flushes and the rank of 1
        # since theyit is the best hand in poker
        # rank 1 = Royal Flush!
        rank = 1
        for sf in straight_flushes:
            prime_product = Card.prime_product_from_rankbits(sf)
            self.flush_lookup[prime_product] = rank
            rank += 1

        # we start the counting for flushes on max full house, which
        # is the worst rank that a full house can have (2,2,2,3,3)
        rank = LookupTable.MAX_FULL_HOUSE + 1
        for f in flushes:
            prime_product = Card.prime_product_from_rankbits(f)
            self.flush_lookup[prime_product] = rank
            rank += 1

        # we can reuse these bit sequences for straights
        # and high cards since they are inherently related
        # and differ only by context
        self.straight_and_highcards(straight_flushes, flushes)

    def straight_and_highcards(self, straights, highcards):
        """
        Unique five card sets. Straights and highcards.

        Reuses bit sequences from flush calculations.
        """
        rank = LookupTable.MAX_FLUSH + 1

        for s in straights:
            prime_product = Card.prime_product_from_rankbits(s)
            self.unsuited_lookup[prime_product] = rank
            rank += 1

        rank = LookupTable.MAX_PAIR + 1
        for h in highcards:
            prime_product = Card.prime_product_from_rankbits(h)
            self.unsuited_lookup[prime_product] = rank
            rank += 1

    def multiples(self):
        """
        Pair, Two Pair, Three of a Kind, Full House, and 4 of a Kind.
        """
        backwards_ranks = list(range(len(Card.INT_RANKS) - 1, -1, -1))

        # 1) Four of a Kind
        rank = LookupTable.MAX_STRAIGHT_FLUSH + 1

        # for each choice of a set of four rank
        for i in backwards_ranks:

            # and for each possible kicker rank
            kickers = backwards_ranks[:]
            kickers.remove(i)
            for k in kickers:
                product = Card.PRIMES[i]**4 * Card.PRIMES[k]
                self.unsuited_lookup[product] = rank
                rank += 1

        # 2) Full House
        rank = LookupTable.MAX_FOUR_OF_A_KIND + 1

        # for each three of a kind
        for i in backwards_ranks:

            # and for each choice of pair rank
            pairranks = backwards_ranks[:]
            pairranks.remove(i)
            for pr in pairranks:
                product = Card.PRIMES[i]**3 * Card.PRIMES[pr]**2
                self.unsuited_lookup[product] = rank
                rank += 1

        # 3) Three of a Kind
        rank = LookupTable.MAX_STRAIGHT + 1

        # pick three of one rank
        for r in backwards_ranks:

            kickers = backwards_ranks[:]
            kickers.remove(r)
            gen = itertools.combinations(kickers, 2)

            for kickers in gen:

                c1, c2 = kickers
                product = Card.PRIMES[r]**3 * Card.PRIMES[c1] * Card.PRIMES[c2]
                self.unsuited_lookup[product] = rank
                rank += 1

        # 4) Two Pair
        rank = LookupTable.MAX_THREE_OF_A_KIND + 1

        tpgen = itertools.combinations(backwards_ranks, 2)
        for tp in tpgen:

            pair1, pair2 = tp
            kickers = backwards_ranks[:]
            kickers.remove(pair1)
            kickers.remove(pair2)
            for kicker in kickers:

                product = Card.PRIMES[pair1]**2 * Card.PRIMES[pair2]**2 * Card.PRIMES[kicker]
                self.unsuited_lookup[product] = rank
                rank += 1

        # 5) Pair
        rank = LookupTable.MAX_TWO_PAIR + 1

        # choose a pair
        for pairrank in backwards_ranks:

            kickers = backwards_ranks[:]
            kickers.remove(pairrank)
            kgen = itertools.combinations(kickers, 3)

            for kickers in kgen:

                k1, k2, k3 = kickers
                product = Card.PRIMES[pairrank]**2 * Card.PRIMES[k1]                         * Card.PRIMES[k2] * Card.PRIMES[k3]
                self.unsuited_lookup[product] = rank
                rank += 1

    def write_table_to_disk(self, table, filepath):
        """
        Writes lookup table to disk
        """
        with open(filepath, 'w') as f:
            for prime_prod, rank in table.items():
                f.write(str(prime_prod) +","+ str(rank) + '\n')

    def get_lexographically_next_bit_sequence(self, bits):
        """
        Bit hack from here:
        http://www-graphics.stanford.edu/~seander/bithacks.html#NextBitPermutation

        Generator even does this in poker order rank
        so no need to sort when done! Perfect.
        """
        t = (bits | (bits - 1)) + 1
        next = t | (( int((t & -t) / (bits & -bits)) >> 1) - 1)
        yield next
        while True:
            t = (next | (next - 1)) + 1
            next = t | (( int((t & -t) / (next & -next)) >> 1) - 1)
            yield next

class Evaluator(object):
    """
    Evaluates hand strengths using a variant of Cactus Kev's algorithm:
    http://www.suffecool.net/poker/evaluator.html

    I make considerable optimizations in terms of speed and memory usage, 
    in fact the lookup table generation can be done in under a second and 
    consequent evaluations are very fast. Won't beat C, but very fast as 
    all calculations are done with bit arithmetic and table lookups. 
    """

    def __init__(self):

        self.table = LookupTable()
        
        self.hand_size_map = {
            5 : self._five,
            6 : self._six,
            7 : self._seven
        }

    def evaluate(self, handCards, tableCards):
        """
        This is the function that the user calls to get a hand rank. 

        Supports empty board, etc very flexible. No input validation 
        because that's cycles!
        """
        cards = handCards + tableCards
        return self.hand_size_map[len(cards)](cards)

    def _five(self, cards):
        """
        Performs an evalution given cards in integer form, mapping them to
        a rank in the range [1, 7462], with lower ranks being more powerful.

        Variant of Cactus Kev's 5 card evaluator, though I saved a lot of memory
        space using a hash table and condensing some of the calculations. 
        """
        # if flush
        if cards[0] & cards[1] & cards[2] & cards[3] & cards[4] & 0xF000:
            handOR = (cards[0] | cards[1] | cards[2] | cards[3] | cards[4]) >> 16
            prime = Card.prime_product_from_rankbits(handOR)
            return self.table.flush_lookup[prime]

        # otherwise
        else:
            prime = Card.prime_product_from_hand(cards)
            return self.table.unsuited_lookup[prime]

    def _six(self, cards):
        """
        Performs five_card_eval() on all (6 choose 5) = 6 subsets
        of 5 cards in the set of 6 to determine the best ranking, 
        and returns this ranking.
        """
        minimum = LookupTable.MAX_HIGH_CARD

        all5cardcombobs = itertools.combinations(cards, 5)
        for combo in all5cardcombobs:

            score = self._five(combo)
            if score < minimum:
                minimum = score

        return minimum

    def _seven(self, cards):
        """
        Performs five_card_eval() on all (7 choose 5) = 21 subsets
        of 5 cards in the set of 7 to determine the best ranking, 
        and returns this ranking.
        """
        minimum = LookupTable.MAX_HIGH_CARD

        all5cardcombobs = itertools.combinations(cards, 5)
        for combo in all5cardcombobs:
            
            score = self._five(combo)
            if score < minimum:
                minimum = score

        return minimum

    def get_rank_class(self, hr):
        """
        Returns the class of hand given the hand hand_rank
        returned from evaluate. 
        """
        if hr >= 0 and hr < LookupTable.MAX_STRAIGHT_FLUSH:
            return LookupTable.MAX_TO_RANK_CLASS[LookupTable.MAX_STRAIGHT_FLUSH]
        elif hr <= LookupTable.MAX_FOUR_OF_A_KIND:
            return LookupTable.MAX_TO_RANK_CLASS[LookupTable.MAX_FOUR_OF_A_KIND]
        elif hr <= LookupTable.MAX_FULL_HOUSE:
            return LookupTable.MAX_TO_RANK_CLASS[LookupTable.MAX_FULL_HOUSE]
        elif hr <= LookupTable.MAX_FLUSH:
            return LookupTable.MAX_TO_RANK_CLASS[LookupTable.MAX_FLUSH]
        elif hr <= LookupTable.MAX_STRAIGHT:
            return LookupTable.MAX_TO_RANK_CLASS[LookupTable.MAX_STRAIGHT]
        elif hr <= LookupTable.MAX_THREE_OF_A_KIND:
            return LookupTable.MAX_TO_RANK_CLASS[LookupTable.MAX_THREE_OF_A_KIND]
        elif hr <= LookupTable.MAX_TWO_PAIR:
            return LookupTable.MAX_TO_RANK_CLASS[LookupTable.MAX_TWO_PAIR]
        elif hr <= LookupTable.MAX_PAIR:
            return LookupTable.MAX_TO_RANK_CLASS[LookupTable.MAX_PAIR]
        elif hr <= LookupTable.MAX_HIGH_CARD:
            return LookupTable.MAX_TO_RANK_CLASS[LookupTable.MAX_HIGH_CARD]
        else:
            raise Exception("Inavlid hand rank, cannot return rank class")

    def class_to_string(self, class_int):
        """
        Converts the integer class hand score into a human-readable string.
        """
        return LookupTable.RANK_CLASS_TO_STRING[class_int]

    def get_five_card_rank_percentage(self, hand_rank):
        """
        Scales the hand rank score to the [0.0, 1.0] range.
        """
        return float(hand_rank) / float(LookupTable.MAX_HIGH_CARD)

    def hand_summary(self, board, hands):
        """
        Gives a sumamry of the hand with ranks as time proceeds. 

        Requires that the board is in chronological order for the 
        analysis to make sense.
        """

        assert len(board) == 5, "Invalid board length"
        for hand in hands:
            assert len(hand) == 2, "Inavlid hand length"

        line_length = 10
        stages = ["FLOP", "TURN", "RIVER"]

        for i in range(len(stages)):
            line = ("=" * line_length) + " %s " + ("=" * line_length) 
            print(line % stages[i])
            
            best_rank = 7463  # rank one worse than worst hand
            winners = []
            for player, hand in enumerate(hands):

                # evaluate current board position
                rank = self.evaluate(hand, board[:(i + 3)])
                rank_class = self.get_rank_class(rank)
                class_string = self.class_to_string(rank_class)
                percentage = 1.0 - self.get_five_card_rank_percentage(rank)  # higher better here
                print("Player %d hand = %s, percentage rank among all hands = %f" % (
                    player + 1, class_string, percentage))

                # detect winner
                if rank == best_rank:
                    winners.append(player)
                    best_rank = rank
                elif rank < best_rank:
                    winners = [player]
                    best_rank = rank

            # if we're not on the river
            if i != stages.index("RIVER"):
                if len(winners) == 1:
                    print("Player %d hand is currently winning.\n" % (winners[0] + 1,))
                else:
                    print("Players %s are tied for the lead.\n" % [x + 1 for x in winners])

            # otherwise on all other streets
            else:
                print()
                print(("=" * line_length) + " HAND OVER " + ("=" * line_length)) 
                if len(winners) == 1:
                    print("Player %d is the winner with a %s\n" % (winners[0] + 1, 
                        self.class_to_string(self.get_rank_class(self.evaluate(hands[winners[0]], board)))))
                else:
                    print("Players %s tied for the win with a %s\n" % (winners, 
                        self.class_to_string(self.get_rank_class(self.evaluate(hands[winners[0]], board)))))


# In[ ]:



