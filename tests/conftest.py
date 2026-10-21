import os

# default OpenBLAS threading makes the dense 1024^2 eigendecompositions ~100x slower
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
