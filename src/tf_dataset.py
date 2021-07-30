# reads voxels
from __future__ import division, print_function, absolute_import

import numpy as np
import sys
import math
import os
import timeit
import argparse
import pickle as pkl

import sklearn.utils
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# one chunk
class SingleChunk:
    def __init__(self, path='', batch_size=32):
        self.path = path
        self.batch_size = batch_size
        if path.endswith('.npy'):
            self.allData = np.load(self.path)
        else:
            self.allData = np.loadtxt(self.path, dtype="float")

        sklearn.utils.check_array(self.allData, ensure_2d=False)

        # if the dataset is 1D, like dosage features, make it 2D
        if len(self.allData.shape) == 1:
            self.allData = np.reshape(self.allData, (-1, 1))

        self.numSamples = self.allData.shape[0]
        self.numFeatures = self.allData.shape[1]

        self.currentIndex = 0

    # will try to return the amount. might be < amount
    def get_amount(self, amount):
        loop = False
        if self.currentIndex+amount >= self.numSamples:
            result = self.allData[self.currentIndex:]
        else:
            result = self.allData[self.currentIndex: \
                        self.currentIndex+amount]

        self.currentIndex += amount
        return result

    def reset(self):
        self.currentIndex = 0

    def randomize(self):
        np.random.shuffle(self.allData)

    def all(self):
        return self.allData

    def set_all(self, mat):
        self.allData = mat

    def is_empty(self):
        return self.currentIndex >= self.numSamples

# one label chunk
class SingleChunkLabel(SingleChunk):
    def __init__(self, path='', batch_size=32):
        self.path = path
        self.batch_size = batch_size
        if path.endswith('.npy'):
            self.allData = np.load(self.path)
        else:
            self.allData = np.loadtxt(self.path, dtype="float")

        sklearn.utils.check_array(self.allData, ensure_2d=False)

        self.numSamples = self.allData.shape[0]
        # regression labels
        self.numClasses = 1

        self.currentIndex = 0

# one label chunk
class KeyedSingleChunkLabel(SingleChunk):
    def __init__(self, path='', batch_size=32):
        self.path = path
        self.batch_size = batch_size
        if path.endswith('.npy'):
            self.allData = np.load(self.path)
        else:
            self.allData = np.loadtxt(self.path, dtype="float")

        sklearn.utils.check_array(self.allData, ensure_2d=False)

        self.numSamples = self.allData.shape[0]
        self.keys = self.allData[:,0]
        self.allData = self.allData[:,1]
        # regression labels
        self.numClasses = 1

        self.currentIndex = 0

    def randomize(self):
        rng_state = np.random.get_state()
        np.random.shuffle(self.allData)
        np.random.set_state(rng_state)
        np.random.shuffle(self.keys)

# takes in multiple chunks. iterates through them all. min size 1
class ChunkData(object):
    def __init__(self, paths=[], batch_size=32, start_chunk=0, preprocessor=None):
        self.paths = paths
        self.batch_size = batch_size

        self.chunk_index = start_chunk - 1
        self.iterate_file()
        self.numFeatures = self.current_set.numFeatures
        self.num_chunks = len(paths)

        self.load_preprocessor(preprocessor)

    #idempotent, wont iterate beyond the end
    def iterate_file(self):
        if self.chunk_index+1 >= len(self.paths):
            return True

        self.chunk_index += 1
        self.current_set = SingleChunk(self.paths[self.chunk_index], \
                            batch_size=self.batch_size)
        self.allData = self.current_set.allData

        return False

    # must return batch_size samples. will repeat across chunks until done
    def get_next_batch(self):
        # must always return some data. can't be empty
        data = self.current_set.get_amount(self.batch_size)
        looped = False
        while data.shape[0] < self.batch_size:
            # current_set must be empty. either reset or iterate
            if self.is_last_chunk():
                self.reset()
                looped = True
            else:
                self.iterate_file()
            new_data = self.current_set.get_amount(self.batch_size-data.shape[0])
            data = np.concatenate([data, new_data], axis=0)

        if self.current_set.is_empty():
            if self.is_last_chunk():
                self.reset()
                looped = True
            else:
                # just need to iterate to next file
                self.iterate_file()

        return data, looped

    def get_onetime_batch(self):
        if self.current_set.is_empty() and self.is_last_chunk():
            return np.array([]), True

        data = None
        empty = False
        while data is None or data.shape[0] < self.batch_size:
            if data is None:
                data = self.current_set.get_amount(self.batch_size)
            else:
                new_data = self.current_set.get_amount(self.batch_size-data.shape[0])
                data = np.concatenate([data, new_data], axis=0)

            if self.current_set.is_empty():
                if self.is_last_chunk():
                    # totally out of data
                    empty = True
                    break
                else:
                    # just need to iterate to next file
                    self.iterate_file()

        return data, empty

    def reset(self):
        self.chunk_index = -1 # this gets set to 0 in self.iterate_file

        self.iterate_file()

    def randomize(self):
        self.current_set.randomize()

    def all(self):
        return self.current_set.allData # definitely can't do this, so we cheat

    def set_all(self, mat):
        self.current_set.set_all(mat)

    def is_last_chunk(self):
        return self.chunk_index == len(self.paths)-1

    def current_chunk_size(self):
        return self.current_set.numSamples

    def is_empty(self):
        return self.current_set.is_empty()

    def get_preprocessor(self):
        return None

    def load_preprocessor(self, preprocessor):
        pass

# special class for dealing with labels.
class ChunkDataLabel(ChunkData):
    def __init__(self, paths=[], batch_size=32, start_chunk=0, preprocessor=None):
        self.paths = paths
        self.num_chunks = len(paths)
        self.batch_size = batch_size

        self.chunk_index = start_chunk - 1
        self.iterate_file()

        self.init_classes(preprocessor)

    def iterate_file(self):
        if self.chunk_index+1 >= len(self.paths):
            return True

        self.chunk_index += 1
        self.current_set = SingleChunkLabel(self.paths[self.chunk_index], \
                            batch_size=self.batch_size)
        return False

    def init_classes(self, preprocessor):
        path = self.paths[0]
        if '.y.' in path or '.lab.' in path or path.endswith('.y'):
            if preprocessor is None:
                # loop through all labels and convert to one hot encoding
                labels = []
                loop = False
                while not loop:
                    l, loop = super(ChunkDataLabel, self).get_onetime_batch()
                    labels.append(l)
                self.reset()

                # reshape to something sklearn likes
                all_labels = np.concatenate(labels).reshape(-1, 1)
                preprocessor = OneHotEncoder(sparse=False)
                preprocessor.fit(all_labels)

            self.load_preprocessor(preprocessor)

        elif path.endswith('.r') or '.r.' in path or \
                path.endswith('._y.csv') or path.endswith('._y_fake.csv'):
            # regression labels
            self.numClasses = 1
            self.encoder = None

    def get_preprocessor(self):
        # in this case, the preprocessor is an OneHotEncoder or None
        return self.encoder

    def load_preprocessor(self, preprocessor):
        # if the preprocessor is None, then it's regression values
        # else it's a OneHotEncoder
        if preprocessor is None:
            self.encoder = None
            self.numClasses = 1
        else:
            self.encoder = preprocessor
            self.numClasses = self.encoder.n_values_[0]

    def get_next_batch(self):
        labels, loop = super(ChunkDataLabel, self).get_next_batch()

        if self.encoder is None:
            return labels, loop
        else:
            labels = labels.reshape(-1, 1)
            transformed = self.encoder.transform(labels)

            return transformed, loop

    def get_onetime_batch(self):
        labels, loop = super(ChunkDataLabel, self).get_onetime_batch()

        if self.encoder is None:
            return labels, loop
        else:
            labels = labels.reshape(-1, 1)
            transformed = self.encoder.transform(labels)

            return transformed, loop

    def inverse_transform(self, labels):
        if self.encoder is None:
            return labels
        else:
            return self.encoder.inverse_transform(labels).reshape(labels.shape[0],)

# special class for dealing with labels.
class KeyedChunkDataLabel(ChunkDataLabel):
    def iterate_file(self):
        if self.chunk_index+1 >= len(self.paths):
            return True

        self.chunk_index += 1
        self.current_set = KeyedSingleChunkLabel(self.paths[self.chunk_index], \
                            batch_size=self.batch_size)
        self.allData = self.current_set.allData
        self.keys = self.current_set.keys

        return False

# keeps several ChunkData and ChunkDataLabel objects aligned
class ChunkGroup(object):
    def __init__(self, pathFeatures=[['']], pathLabels=[], batch_size=32, shuffle=True, \
                preprocessing_fn='no_preprocessors.pkl'):
        print('features', pathFeatures)
        print('labels', pathLabels)

        # check all features have some # of chunks and labels
        num_chunks = len(pathFeatures[0])
        assert all([len(l) == num_chunks for l in pathFeatures+[pathLabels]])

        self.pathFeatures = pathFeatures
        self.pathLabels = pathLabels
        self.batch_size = batch_size

        # supports multiple feature vectors per sample
        print("no progress file found. starting over")
        self.chunk_index = 0

        feat_pre, label_pre = self.load_preprocessing(preprocessing_fn)

        self.features = [ChunkData(path_feature, self.batch_size, start_chunk=self.chunk_index, preprocessor=feat_pre[i]) \
                            for i,path_feature in enumerate(self.pathFeatures)]

        # moved this to a function to allow for easy implementation of keyed labels
        self.load_labels(label_pre)

        self.save_preprocessing(preprocessing_fn)

        # randomize the first chunk
        self.shuffle = shuffle
        if self.shuffle:
            self.randomize()

        # numFeatures is the sum of all feature sets
        total_features = 0
        for feat in self.features:
            total_features += feat.numFeatures
        self.numFeatures = total_features

        # make sure all files are aligned
        # same number of chunks for each type of input
        self.num_chunks = self.features[0].num_chunks
        assert all([feat.num_chunks == self.num_chunks for feat in self.features])
        if not self.labels is None:
            assert self.num_chunks == self.labels.num_chunks, \
                "num_chunks %d labels.num_chunks %d" % (self.num_chunks, self.labels.num_chunks)

        # make sure all current chunks are the same size
        self.current_chunk_size_check()

    def save_preprocessing(self, preprocessing_fn):
        # input:
        # preprocessing_fn: string file name
        # output: None

        feat_pre = [cd.get_preprocessor() for cd in self.features]
        if self.labels is None:
            label_pre = None
        else:
            label_pre = self.labels.get_preprocessor()

        pkl.dump({'feat_pre':feat_pre, 'label_pre':label_pre}, open(preprocessing_fn, 'wb'))

    def load_preprocessing(self, preprocessing_fn):
        # input:
        # preprocessing_fn: string file name
        # output:
        # feat_pre: list of preprocessing objects for features in order of application to ChunkData
        # label_pre: preprocessing object for label

        if os.path.exists(preprocessing_fn):
            dat = pkl.load(open(preprocessing_fn, 'rb'))

            return dat['feat_pre'], dat['label_pre']
        else:
            return [None]*len(self.pathFeatures), None

    def load_labels(self, label_pre):
        if len(self.pathLabels) > 0:
            self.labels = ChunkDataLabel(self.pathLabels, self.batch_size, 
                start_chunk=self.chunk_index, preprocessor=label_pre)
            self.numClasses = self.labels.numClasses
        else:
            self.labels = None
            self.numClasses = 0

    def current_chunk_size_check(self):
        # make sure all current chunks have the same size
        if not self.labels is None:
            all_files = self.features + [self.labels]
        else:
            all_files = self.features

        corret_size = all_files[0].current_chunk_size()
        assert all([corret_size == a_file.current_chunk_size() for a_file in all_files])

    def get_next_batch(self):
        # check chunk alignment
        files = self.features + [self.labels]
        for f in files:
            assert self.chunk_index == f.chunk_index

        # collect features from all feature sets
        fs = []
        loopB = None
        for feats in self.features:
            f, loopB = feats.get_next_batch()

            fs.append(f)

        if not self.labels is None:
            ls, loopA = self.labels.get_next_batch()
        else:
            ls = np.array([])
            loopA = loopB

        assert loopA == loopB

        data = tuple(fs) + (ls, loopA)

        # did we start a new chunk?
        new_chunk_index = self.features[0].chunk_index
        if new_chunk_index != self.chunk_index:
            # make sure all current chunks are the same size
            self.current_chunk_size_check()

            if self.shuffle:
                self.randomize()
            self.chunk_index = new_chunk_index

        return data

    def get_onetime_batch(self):
        # collect features from all feature sets
        fs = []
        loopB = None
        for feats in self.features:
            f, loopB = feats.get_onetime_batch()
            fs.append(f)

        if not self.labels is None:
            ls, loopA = self.labels.get_onetime_batch()
        else:
            ls = np.array([])
            loopA = loopB

        assert loopA == loopB

        data = tuple(fs) + (ls, loopA)

        # did we start a new chunk?
        new_chunk_index = self.features[0].chunk_index
        if new_chunk_index != self.chunk_index:
            # make sure all current chunks are the same size
            self.current_chunk_size_check()
            self.chunk_index = new_chunk_index

            if self.shuffle:
                self.randomize()

        return data

    def get_all_label_names(self):
        # returns the path to the label file
        if not self.is_empty():
            if len(self.pathLabels) == 0 or "NCIPDM" in self.pathFeatures[0][0]:
                # hard code this to use the dosage as the name if no labels
                # or for NCIPDM dataset
                return self.pathFeatures[2]
            else:
                return self.pathLabels
        else:
            # if you're out of samples, return ''
            return ''

    def current_label_name(self):
        # returns the path to the label file that will produce the NEXT batch
        if not self.is_empty():
            if len(self.pathLabels) == 0 or "NCIPDM" in self.pathFeatures[0][0]:
                # hard code this to use the dosage as the name if no labels
                # or for NCIPDM dataset
                return self.pathFeatures[2][self.chunk_index]
            else:
                #print("len pathLabels", len(self.pathLabels), "chunk_index", self.chunk_index)
                return self.pathLabels[self.chunk_index]
        else:
            # if you're out of samples, return ''
            return ''

    def is_empty(self):
        return self.features[0].is_empty()

    def reset(self):
        for feat in self.features:
            feat.reset()

        if not self.labels is None:
            self.labels.reset()

        self.chunk_index = 0

    def randomize(self):
        rng_state = np.random.get_state()
        for cd in self.features+[self.labels]:
            cd.randomize()
            np.random.set_state(rng_state)

        # with out this call, every chunk will always be randomized the same way
        # in other words each epoch will have the same order
        np.random.shuffle(np.zeros(4))

    def all(self):
        self.reset()
        allFeatures = [feat.all() for feat in self.features]

        if not self.labels is None:
            allLabels = self.labels.all()
        else:
            allLabels = np.array([])

        return tuple(allFeatures) + (allLabels,)

    def inverse_transform(self, labels):
        if self.labels is None:
            return labels
        else:
            return self.labels.inverse_transform(labels)

class KeyedChunkGroup(ChunkGroup):
    def load_labels(self, labels_pre):
        if len(self.pathLabels) > 0:
            self.labels = KeyedChunkDataLabel(self.pathLabels, self.batch_size, 
                start_chunk=self.chunk_index, preprocessor=labels_pre)
            self.numClasses = self.labels.numClasses
        else:
            self.labels = None
            self.numClasses = 0

class VariableSet:
    def __init__(self, checkpoint):
        self.checkpoint = checkpoint
        self.variables = set()

        with open(checkpoint.replace('_ratchet', '')+'.vars') as variableFile:
            print("opened the vars file")
            for l in variableFile:
                self.variables.add(l.strip())

    @staticmethod
    def writeFile(variables, filename):
    # we add .vars to the end of the filename given
        filename += '.vars'
        with open(filename, 'w') as varFile:
            variableNames = [t.name for t in variables]
            for name in variableNames:
                varFile.write(name+"\n")

    def intersection(self, variables):
        result = []
        print("input variables", [v.name for v in variables])
        for v in variables:
            if v.name in self.variables:
                result.append(v)

        return result

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-X', '--trainx', nargs='+', default=[''], action='store', help='path to training samples')
    parser.add_argument('-Z', '--trainz', nargs='+', default=[''], action='store', help='secondary training feature samples')
    parser.add_argument('--traind', nargs='+', default=[''], action='store', help='tertiary trianing feature samples')
    parser.add_argument('-y', '--trainy', nargs='+', default=[''], action='store', help='path to training labels')

    args = parser.parse_args()

    print(args)

    return args

def test_small_samples():
    args = parseArgs()

    # make sure input test files have < 100 samples
    cg = ChunkGroup(pathFeatures=[args.trainx, args.trainz], pathLabels=args.trainy, 
                        batch_size=30)

    loop = False
    start = timeit.default_timer()
    for i in range(9):
        data = cg.get_onetime_batch()
        loop = data[-1]

        for d in data[:-1]:
            print(d.shape)

    stop = timeit.default_timer()
    print("runtime time for epoch", stop-start)

def loop_samples():
    args = parseArgs()

    # make sure input test files have < 100 samples
    cg = ChunkGroup(pathFeatures=[args.trainx, args.trainz], pathLabels=args.trainy, 
                        batch_size=30000)

    loop = False
    start = timeit.default_timer()
    while not loop:
        data = cg.get_onetime_batch()
        loop = data[-1]

        for d in data[:-1]:
            print(d.shape)

    stop = timeit.default_timer()
    print("runtime time for epoch", stop-start)

    print("reset")
    cg.reset()

    loop = False
    start = timeit.default_timer()
    while not loop:
        data = cg.get_onetime_batch()
        loop = data[-1]

        for d in data[:-1]:
            print(d.shape)

    stop = timeit.default_timer()
    print("runtime time for epoch", stop-start)

def label_test():
    args = parseArgs()

    cg = ChunkGroup(pathFeatures=[args.trainx], pathLabels=args.trainy, batch_size=4, preprocessing_fn='regressor_encoder.pkl')

    loop = False
    while not loop:
        x, y, loop = cg.get_next_batch()
        print('loop:',loop)
        print('x:',x)
        print('y:',y)
        print('rev:',cg.inverse_transform(y))

def preprocessor_test():
    # run this in /g/g19/he6/pilot1/refactor/onehot_test
    cg1 = ChunkGroup(pathFeatures=[['X1.npy', 'X2.npy']], pathLabels=['y1.y', 'y2.y'], batch_size=4, preprocessing_fn='encoder.pkl')

    loop = False
    while not loop:
        x, y, loop = cg1.get_next_batch()
        print('loop:',loop)
        print('x:',x)
        print('y:',y)

    cg2 = ChunkGroup(pathFeatures=[['X2.npy']], pathLabels=['y2.y'], batch_size=4, preprocessing_fn='encoder.pkl')

    loop = False
    while not loop:
        x, y, loop = cg2.get_next_batch()
        print('loop:',loop)
        print('x:',x)
        print('y:',y)

if __name__ == '__main__':
    #preprocessor_test()
    label_test()



