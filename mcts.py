##############################################################################
# Written by: Louie Liu and Kian Rossitto
# Date: 4/12/20
# Purpose: These classes make up the implementation of MCTS for playing go.
#############################################################################

import math
import collections
import numpy as np
import time
import threading
from concurrent.futures import Future

import go
from utils import *



class DummyNode:
    """
    This is a node that is the node above the root node. Needed to avoid
    certain special conditions.
    """

    def __init__(self):
        self.parent = None
        self.child_N = collections.defaultdict(float)
        self.child_W = collections.defaultdict(float)
##########################################################################
# NOTE: These constants were discovered through viewing some examples
# as well as trial and error


# Exploration constant
c_EXPLORE = 1.34
# Dirichlet noise alpha parameter
D_NOISE_ALPHA = 0.03
# Board Size
B_SIZE = 9
# Max number of steps before we selection action with
# highest action probability. This is around the # of
# plays currently made. Thus, we want slightly more than
# the number of points on the board because capturing
# can extended the number of moves played in a game.
MAX_STEPS = B_SIZE**2*1.5
# Found through trial and error
RES_THRESH = -0.75
##########################################################################

class MCTSNode:
    """
    A node in the Monte-Carlo search tree.
    """
    def __init__(self, status, move=None, parent=None):
        """
        :param status: Status for the node to hold
        :param move: The move that came from the parent node
        :param parent: parent node
        """

        if parent is None:
            parent = DummyNode()

        self.parent = parent
        self.move = move
        self.status = status
        # This determine if the node is a leaf node. Leaf nodes have not been expanded
        self.is_expanded = False
        self.child_N = np.zeros([B_SIZE*B_SIZE+1], dtype=np.float32)
        self.child_W = np.zeros([B_SIZE*B_SIZE+1], dtype=np.float32)
        self.child_prior = np.zeros([B_SIZE*B_SIZE+1], dtype=np.float32)
        self.children = {}

    def get_move_num(self, tup=None):
        """
        Get the number version of the the current move. If
        conversion is required, provide the tuple to convert.
        :param tup: The tuple to convert to a number
        :return: The move number as an integer.
        """
        if tup is None:
            tup = self.move

        if tup is None: return 81
        return tup[0] * 9 + tup[1]

    def get_move_tuple(self, num):
        """
        Get the tuple version of the the current move. If
        conversion is required, provide the int to convert.
        :param num: The number to convert to a tuple
        :return: The move number as a tuple.
        """
        if num is None:
            num = self.move

        if num==81: return None
        return num//9, num%9

    @property
    def N(self):
        """
        Returns current visit count of node

        """
        return self.parent.child_N[self.move]

    @N.setter
    def N(self, value):
        self.parent.child_N[self.move] = value

    @property
    def W(self):
        """
        Returns current total value of node
        """
        return self.parent.child_W[self.move]

    @W.setter
    def W(self, value):
        self.parent.child_W[self.move] = value

    @property
    def Q(self):
        """
        Returns the current action value of the node.
        """
        return self.W / (1 + self.N)

    @property
    def child_Q(self):
        return self.child_W / (1 + self.child_N)

    @property
    def child_U(self):
        return (c_EXPLORE * math.sqrt(1 + self.N) *
                self.child_prior / (1 + self.child_N))

    @property
    def child_action_score(self):
        """
        Action_Score(s, a) = Q(s, a) + U(s, a) as in Alphago paper. A high value
        means the node should be traversed.
        """
        return self.child_Q + self.child_U

    def selection(self):
        """
        Traverses the tree from current node until it finds a leaf node.
        Nodes selected according to child_action_score. the leaf node is then
        expanded through adding a MCTSNode.

        :return: Expanded leaf MCTSNode
        """
        curr = self
        while True:
            # We encountered a leaf node
            if not curr.is_expanded:
                break
            curr.N += 1

            # If there are no more legal moves, pass.
            if (len(curr.status.recent) > 1
                    and curr.status.recent[-1] is None
                    and curr.child_N[-1] == 0):
                curr = curr.expansion(81)
                continue

            best_move = np.argmax(curr.child_action_score)
            # Expand current node
            curr = curr.expansion(best_move)
        return curr

    def expansion(self, move):
        """
        Expansion step of MCTS. Adds a child node for the action if it doesn't
        exist, then returns it.

        :param action: action for current status which leads to child node
        :return: Child MCTSNode
        """
        if move not in self.children:
            new_status = self.status.copy()
            new_status.play_move(self.get_move_tuple(move))
            self.children[move] = MCTSNode(new_status, move, parent=self)
        return self.children[move]

    def backpropagation(self, value, up_to):
        """
        backpropagate a value estimation to root node once the game is over.
        :param value: Value estimate to backpropagate
        :param up_to: The node to backpropagate to.
        """
        self.W += value
        if self.parent is None or self is up_to:
            return
        self.parent.backpropagation(value, up_to)

    def pre_backpropagation(self, value, up_to):
        """
        Certain steps that are needed prior to backpropagating including
        adding in noise, expanding leaf nodes and adjusting the scores.
        :param value: Value estimate to backpropagate
        :param up_to: The node to backpropagate to.
        """
        self.is_expanded = True
        self.add_noise()
        self.N += 1
        self.child_W = np.ones([B_SIZE*B_SIZE+1], dtype=np.float32)

        if self.parent is None or self is up_to:
            return
        self.parent.backpropagation(value, up_to)


    def is_done(self):
        """
        Checking if the game is over.
        :return: True if game is over. False if not.
        """
        return self.status.is_game_over() or self.status.n > MAX_STEPS

    def add_noise(self):
        """
        Adding noise to the children via AlphaGo paper
        """
        dirch = np.random.dirichlet([D_NOISE_ALPHA] * 82)
        self.child_prior = self.child_prior * 0.76 + dirch * 0.25

def call_with_future(fn, future, args, kwargs):
    try:
        result = fn(*args, **kwargs)
        future.set_result(result)
    except Exception as exc:
        future.set_exception(exc)
def threaded(fn):
    """
    Function to help class functions become threaded.
    :param fn: The function to thread
    :return: the thread
    """
    def wrapper(*args, **kwargs):
        future = Future()
        threading.Thread(target=call_with_future, args=(fn, future, args, kwargs)).start()
        return future
    return wrapper

class MCTSOpponent:
    """
    An opponent using MCTS.
    """

    def __init__(self, status=None, sec_per_move=5, search_n=100, player_mode=colormap['white']):
        self.sec_per_move = sec_per_move
        self.root = None
        self.result = 0
        self.search_n = search_n
        self.player_mode = player_mode
        self.status = status
        self.initialize_game(status)

    def reset(self):
        """
        Resets the game state back to an empty board
        """
        self.initialize_game()

    def initialize_game(self, status=None):
        """
        Initializes the game to the defined status
        :param status: The status for the game to be at. If none, becomes an empty board
        """
        if status is None:
            status = go.GoStatus()
        self.root = MCTSNode(status)
        self.status = status
        self.result = 0

    @threaded
    def simulation(self, status):
        """
        Does the simulation part of the MCTS process. Currently is in a simple
        form of randomly choosing a legal move to play for both sides until the
        game is over.
        :return: 1 if a victory, -1 if a loss
        """
        curr_status = status.copy()
        my_move = True
        while not curr_status.is_game_over():
            legal_moves = np.argwhere(curr_status.get_legal_moves())
            legal_moves = legal_moves.reshape(legal_moves.shape[0])
            # Randomly choose a move from the legal moves
            move_choice = self.root.get_move_tuple(np.random.choice(legal_moves))
            # Make my move
            if my_move:
                curr_status.play_move(move_choice)
                my_move = False
            # Make opponent's move
            else:
                curr_status.play_move(move_choice)
                my_move = True
        # Currently computer will always be white.
        return 1 if curr_status.get_score() > 0 else -1

    def tree_search(self):
        """
        Performs a simulation on a selected leaf node. Then backpropagate the result of
        the simulation.
        :return: The leaf node that was expanded
        """
        # Selection and expansion
        leaf = self.root.selection()
        start = time.time()
        if leaf.is_done():
            value = 1 if leaf.status.get_score() > 0 else -1
            # No simulation needed as the game is over, backpropagate
            leaf.backpropagation(value, self.root)
        else:
            # Simulation
            f1 = self.simulation(leaf.status)
            f2 = self.simulation(leaf.status)
            f3 = self.simulation(leaf.status)

            v1 = f1.result()
            v2 = f2.result()
            v3 = f3.result()

            value = v1 + v2 + v3
            leaf.pre_backpropagation(value, self.root)
        print("time for simulation: ", time.time() - start)
        return leaf

    def suggest_move(self):
        """
        Uses tree_search to then update the child_N arrays and then picks the best move.
        :return: The best looking move based on child_N score.
        """
        curr_n = self.root.N
        print("suggesting move")
        while self.root.N < curr_n + self.search_n:
            self.tree_search()

        return self.pick_best_move()

    def pick_best_move(self):
        """
        Looks at the child_N array and chooses a valid move with the highest score.
        :return: Returns the best move
        """
        # A way to get the 10 children with the largest N values sorted in order.
        # This is because there is a bug where the opponent will sometimes choose the spot
        # the player played. This ensures that this does not happen by checking to see
        # if it is a valid move.
        ind = np.argpartition(self.root.child_N, -10)[-10:]
        ind = ind[np.argsort(self.root.child_N[ind])]
        pick = self.root.get_move_tuple(ind[-1])
        while not self.status.is_move_legal(pick):
            ind = ind[:-1]
            pick = self.root.get_move_tuple(ind[-1])

        return pick

    def should_resign(self):
        """
        Whether the MCTS player should resign based on a threshold.

        Note: This particular function was based on the function from
        the GitHub repo VainF/AlphaDoge.
        :return: True if bot should resign, False if no.
        """
        if self.player_mode==colormap['white']:
            return self.root.Q < RES_THRESH
        if self.player_mode==colormap['black']:
            return self.root.Q > -RES_THRESH
        else:
            if self.root.to_play==colormap['white']:
                return self.root.Q < RES_THRESH
            if self.root.to_play==colormap['black']:
                return self.root.Q > -RES_THRESH

    def play_move(self, coord):
        """
        Plays a move on the board, then clears out the root's children as they are
        no longer needed.
        :param coord: The coordinate to play at.
        """
        self.root = self.root.expansion(self.root.get_move_num(coord))
        self.status = self.root.status
        del self.root.parent.children







