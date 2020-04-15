##############################################################################
# Written by: Louie Liu and Kian Rossitto
# Date: 4/12/20
# Purpose: This class uses the MCTS implementation to create a threaded
# opponent MCTS player. This example was based on one found in the project at
# https://github.com/VainF/AlphaDoge
#############################################################################

from PyQt5.QtCore import *

from mcts import MCTSOpponent

class Opponent(QThread):
    tuple_signal = pyqtSignal(tuple)

    def __init__(self, seconds_per_move=5, search_n=100):
        QThread.__init__(self)
        self.player = MCTSOpponent(sec_per_move=5, search_n=100)

    def play_move(self, coord):
        self.player.play_move(coord)

    def reset(self):
        self.player.reset()

    def run(self):
        coord = self.player.suggest_move()
        if coord==None: coord=(-1,-1)
        self.tuple_signal.emit(coord)