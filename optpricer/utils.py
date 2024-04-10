from enum import Enum

class OptPayoff(int, Enum):
    call = 1
    put = -1


class OptType(int, Enum):
    European = 0 # Non-path Dependent, terminal
    American = 1 # Path Dependent
    Asian = 2 # Path Depdendent, averaging across path
    Dummy = 3 # Path Dependent, European (for testing)