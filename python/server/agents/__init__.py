from . import dqn  # registers ("oh_hell", "dqn")
from .registry import load, register, registered

__all__ = ["load", "register", "registered"]
