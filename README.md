# MCTS

A simple implementation of MCTS for playing Go. The simulation is just random playing of valid moves, though adding in a neural network is in the works. 

The Go UI was provided from https://github.com/VainF/AlphaDoge

**Overview**

A module was written containing two classes, MCTS and MCTSOpponent. MCTS implements most of the basic functionality of the MCTS algorithm including selection, expansion, and backpropagation. It also contains values for the children and their action scores, visits, and generally anything else that needs to be kept track of. MCTSOpponent implements MCTS to find the best moves and play them based on # of searches requested per move.

**To Run**

`python main.py` is enough to get this running.

**Thoughts on performance**

Overall, the performance is not great, mostly due to how simulation is performed, which is why adding in a neural network is in the works. However, not adding a neural network initially was done on purpose because the assignment for this class was specifically not to use neural networks. 

Because of the randomness of the simulation, the overall strength was quite weak, from my estimation just slightly better than learning the rules. For reference, I was a 5 Dan amatuer at my strongest while playing Go. Additionally, the time per simulation ranged from 0.05 - 0.75s with an average of 0.3 seconds. This is because I was constrained by the UI that I ended up using. It uses a class called `GoStatus` which contains the board state, valid moves, color of current player, and whatever else is needed to keep track of the board state. The process for simulation is to get the status, choose a random legal move, play it, then loop until the game is over. However, because there is an unknown number of moves needed to get to that point, that is the reason for the long simulation times as well as the high variance. We need to constantly get the updated GoStatus because valid moves change depending on the moves played as well as whether you are black or white.
