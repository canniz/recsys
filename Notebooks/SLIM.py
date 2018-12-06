from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
import importlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse
import random
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfTransformer

TEST_SET_THRESHOLD = 10
TEST_SET_HOLDOUT = 0.2
BEST_ALFA = 0.92

importlib.import_module("utils")

tracks = pd.read_csv('../input/tracks.csv')
train = pd.read_csv('../input/train.csv')
target = pd.read_csv('../input/target_playlists.csv')

icm_csr = utils.build_icm_csr(tracks)
urm_csr = utils.build_urm_csr(train)

urm_csr = utils.build_urm_csr(train)
