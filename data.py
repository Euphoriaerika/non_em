import numpy as np

caseConfig = 1

if caseConfig == 1:
    # configuration constants
    N = 100
    M = 30
    w0 = 7 * 2**0.5 / 12  # frequency of the cosine signal
    y = N / 20  # window width parameter

    # load data from input_data folder
    u1 = np.loadtxt("input_data/u1var5.txt")
    y1 = np.loadtxt("input_data/y1var5.txt")
    y0 = np.loadtxt("input_data/y0var5.txt")
    up = np.loadtxt("input_data/upvar5.txt")
    yp = np.loadtxt("input_data/ypvar5.txt")

if caseConfig == 2:
    # configuration constants
    n = 3
    N = 50

    # load data from input_data folder
    y = np.loadtxt("input_data/yvar5.txt")
