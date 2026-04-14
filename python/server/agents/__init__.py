from . import dqn   # registers ("oh_hell", "dqn")
from . import nfsp  # registers ("oh_hell", "nfsp")
from . import ppo   # registers ("oh_hell", "ppo")
from .registry import load, register, registered

__all__ = ["load", "register", "registered"]
