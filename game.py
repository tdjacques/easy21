from random import randint
import numpy as np

class Easy21(object):
    """
    Setup game of Easy21
    """
    def __init__(self):
        self.new_game()

    def new_game(self):
        self.player_hand = randint(1, 10)
        self.dealer_hand = randint(1, 10)
        self.reward = 0

    def state(self):
        return [self.player_hand, self.dealer_hand]

    def hit(self):
        newCard = randint(1, 10)
        colour = randint(1, 3)
        if colour == 1:
            return -newCard
        else:
            return newCard

    def is_terminal(self): # Check if hand is terminal
        terminal = False
        if self.player_hand > 21 or self.player_hand < 1:
            terminal = True
        elif self.dealer_hand >= 17 or self.dealer_hand < 1:
            terminal = True
        return terminal


    def step(self, action):
        # If player hits
        if action == 1:
            self.player_hand += self.hit()
        # Dealer's turn
        elif action == 0:
            while not self.is_terminal():
                self.dealer_hand += self.hit()
        # Determine score
        if self.is_terminal():
            if self.player_hand > 21 or self.player_hand < 1 or (self.dealer_hand <= 21 and self.dealer_hand > self.player_hand):
                self.reward = -1.
            elif self.dealer_hand > 21 or self.dealer_hand < 1 or self.player_hand > self.dealer_hand:
                self.reward = 1.

            self.player_hand = self.dealer_hand = 0 # bust state

        return [self.state(), self.reward]

    def phi_test(self,i_player,i_dealer,i_action,action):
        """
        Helper function for approximation of state
        """
        test_player = 0 < (self.dealer_hand - 3 * i_dealer) / 4. <= 1
        test_dealer = 0 < (self.player_hand - 3 * i_player) / 6. <= 1
        test_action = action != i_action
        if test_player and test_dealer and test_action:
            return 1
        else: return 0

    def phi(self,action):
        """
        Approximation of state
        """
        return [self.phi_test(i,j,k,action) for k in range(0,2) for j in range(0,3) for i in range(0,6)]


