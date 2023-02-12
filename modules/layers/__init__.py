from .Blocks import *
try:
    from .SparseBlocks import *
except ImportError:
    print('Spconv is supported. You can install the spconv package and set "sparsemiddle" to True to use sparse middle layers.')