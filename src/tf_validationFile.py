from __future__ import division, print_function, absolute_import

import os
import numpy as np
import math
from sklearn.metrics import mean_squared_error

class ValidationFile:
    def __init__(self, filename):
        self.filename = filename
        self.avgName = filename.replace(".csv", ".avg.csv")
        self.output = open(filename, 'a')

        self.ravg = []
        self.navg = []
        self.tavg = []
        self.epochs = []

    def process(self, epoch, original, encode_decode):
        #print("original")
        #print(original)
        #print("encode_decode")
        #print(encode_decode)
        r = []
        n = []
        t = []
        for i, orow in enumerate(original):
            drow = encode_decode[i]
            rms = math.sqrt(mean_squared_error(orow, drow))
            norm = np.linalg.norm(orow-drow)
            tanimoto = tanimotoCoeff(orow, drow)
            r.append(rms)
            n.append(norm)
            t.append(tanimoto)

        rline = ", ".join(map(str, r))
        self.output.write(rline+"\n")
        nline = ", ".join(map(str, n))
        self.output.write(nline+"\n")
        tline = ", ".join(map(str, t))
        self.output.write(tline+"\n")

        self.epochs.append(epoch)
        self.ravg.append(np.mean(r))
        self.navg.append(np.mean(n))
        self.tavg.append(np.mean(t))

    def write(self, epoch, rms, norm, tanimoto):
        #print("printing", epoch, rms, norm, tanimoto)
        self.output.write("{:d}, {:.5f}, {:.5f}, {:.5f}\n".format(epoch, rms, norm, tanimoto))

    def close(self):
        self.output.close()
        if os.path.isfile(self.avgName):
            avgFile = open(self.avgName, 'a')
        else:
            avgFile = open(self.avgName, 'w')
            avgFile.write("epoch, rms, norm, tanimoto\n")

        for i, r in enumerate(self.ravg):
            avgFile.write(str(self.epochs[i])+", "+str(r)+", "+str(self.navg[i])+", "+str(self.tavg[i])+"\n")

        avgFile.close()

def tanimotoCoeff(a, b):
    return np.inner(a,b) / (np.inner(a,a)+np.inner(b,b)-np.inner(a,b))

if __name__ == "__main__":
    testName = "validtest.csv"
    vf = ValidationFile(testName)
    original = np.array([[1, 2, 3], [3, 2, 1]])
    recon = np.array([[1,2,2], [3,2,1]])

    vf.process(1, original, recon)
    vf.close()

    vf2 = ValidationFile(testName)
    original = np.array([[1, 2, 3], [3, 2, 1]])
    recon = np.array([[1,2,2], [3,2,1]])

    vf2.process(1, original, recon)
    vf2.close()