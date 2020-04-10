import numpy as np
import time
import threading
from PyQt5.QtCore import *
from utils import *

from mcts import MCTSOpponent

class Opponent(QThread):
    tuple_signal = pyqtSignal(tuple)

    def __init__(self, seconds_per_move=5, search_n=100):
        QThread.__init__(self)
        self.player = MCTSOpponent(sec_per_move=5, search_n=100)

    def play_move(self, coord):
        self.player.play_move(coord)

    def reset(self):
        self.player.rest()

    def run(self):
        coord = self.player.suggest_move()
        if coord==None: coord=(-1,-1)
        self.tuple_signal.emit(coord)