import pyspiel
from open_spiel.python.bots import uniform_random, human
from open_spiel.python.examples.tic_tac_toe_qlearner import main as tictactoemain
import numpy as np


def main():
    tictactoemain(True)


if __name__ == "__main__":
    main()
