# given pca and some features, test how good reconstruction is
import argparse
from sklearn.externals import joblib
import numpy
from tf_dataset import GDCDataset
from tf_validationFile import ValidationFile

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pca', action='store', help='path to pca')
    parser.add_argument('--features', action='store', help='path to features')
    parser.add_argument('--output', action='store', help='output is csv with relevant information about reconstruction error')

    args = parser.parse_args()

    batch_size = 100
    td = GDCDataset(args.features, batch_size)
    pca = joblib.load(args.pca)
    validationFile = ValidationFile(args.output)

    loop = False
    loopCount = 0
    while not loop:
        X, loop = td.get_next_batch()
        xTrans = pca.transform(X)
        xTransInv = pca.inverse_transform(xTrans)
        validationFile.process(loopCount, X, xTransInv)

        if loopCount == 0:
            print "X shape", X.shape
            print "xTrans shape", xTrans.shape
            print "inverse Trans shape", xTransInv.shape
        loopCount += 1

    validationFile.close()