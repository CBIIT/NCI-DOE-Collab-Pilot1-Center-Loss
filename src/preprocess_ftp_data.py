from os import sep
import pandas as pd
import numpy as np
import argparse
import os

import sklearn.preprocessing as skprep
import sklearn.model_selection as skms

def parse_args():
    '''
    Parse input arguments
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument('combined_rnaseq_data', 
            nargs='?',
            default='../data/ftp_data/combined_rnaseq_data',
            help='path to file downloaded from here: \
                https://ftp.mcs.anl.gov/pub/candle/public/benchmarks/Pilot1/combo/combined_rnaseq_data')

    parser.add_argument('combined_cl_metadata', 
            nargs='?',
            default='../data/ftp_data/combined_cl_metadata',
            help='path to file downloaded from here: \
                https://ftp.mcs.anl.gov/pub/candle/public/benchmarks/Pilot1/combo/combined_cl_metadata')
   
    parser.add_argument('output_dir',
            nargs='?',
            default='../data/processed_ftp_data/',
            help='output directory for processed data')

    return parser.parse_args()

class CombinedRNASeqData:
    def __init__(self, crd_path, ccm_path, min_count_limit=100):
        '''
        Processes FTP RNASEQ dataset

        Arguments:
        crd_path (str): path to combined_rnaseq_data
        ccm_path (str): path to combined_cl_metadata
        min_count_limit (int): minimum class size any class with fewer examples will be
            dropped
        '''
        self.min_count_limit = min_count_limit

        # read combined_rnaseq_data
        self.df = pd.read_csv(crd_path, sep='\t')
        self.feat_cols = self.df.columns[1:]

        # read combined_cl_metadata
        self.cl_df = pd.read_csv(ccm_path, sep='\t')

        # merge rnaseq with combined_cl_metadata
        self.both_df = self.df.merge(self.cl_df, left_on='Sample', 
            right_on='sample_name', how='inner', validate='one_to_one')

        print('num rnaseq samples', len(self.df))
        print('num cell line labels', len(self.cl_df))
        print('num after merge', len(self.both_df))

        # create label mapping
        # class is defined as tumor_site + sample category.
        # discard and classes with fewer than 100 samples
        self.both_df['tissue_class'] =\
            self.both_df[['tumor_site_from_data_src', 'sample_category']]\
                .apply(lambda x: x[0].lower()+' '+x[1], axis=1)

        cats, counts = np.unique(self.both_df['tissue_class'].values, return_counts=True)
        count_dict = {cat:count for cat, count in zip(cats, counts)}
        #print(count_dict)
        print('number of classes with more than %d examples: %d'%\
            (self.min_count_limit, len([1 for c in counts if c>=self.min_count_limit])))
        print('number of classes with fewer than %d examples: %d'%\
            (self.min_count_limit, len([1 for c in counts if c<self.min_count_limit])))

        self.both_df = self.both_df[\
                [count_dict[cat]>=self.min_count_limit for cat in self.both_df['tissue_class'].values]\
            ]
        self.both_df = self.both_df.reset_index(drop=True)

        print('dropped classes with fewer than %d examples'%self.min_count_limit)
        print(len(self.both_df))
        print('expecting around', 12642)

        # encode labels as integers
        self.label_encoder = skprep.LabelEncoder()
        self.label_encoder.fit(self.both_df['tissue_class'].values)
        self.both_df['tissue_class_int'] = \
            self.label_encoder.transform(self.both_df['tissue_class'].values)

        # perform split
        self.train_index, vt_index = next(skms.StratifiedShuffleSplit(n_splits=1, test_size=0.2)\
            .split(np.ones(shape=(len(self.both_df), 2)), self.both_df['tissue_class_int'].values))

        not_train_df = self.both_df.iloc[vt_index] 

        vt_valid_index, vt_test_index = next(skms.StratifiedShuffleSplit(n_splits=1, test_size=0.5)\
            .split(np.ones(shape=(len(not_train_df), 2)), not_train_df['tissue_class_int'].values))

        self.valid_index = np.array(vt_index)[vt_valid_index]
        self.test_index = np.array(vt_index)[vt_test_index]

        self.both_df['subset'] = ['']*len(self.both_df)
        self.both_df.loc[self.both_df.index.isin(self.train_index), 'subset'] = 'train'
        self.both_df.loc[self.both_df.index.isin(self.valid_index), 'subset'] = 'valid'
        self.both_df.loc[self.both_df.index.isin(self.test_index), 'subset'] = 'test'

        self.train_df = self.both_df.iloc[self.train_index]
        self.valid_df = self.both_df.iloc[self.valid_index]
        self.test_df = self.both_df.iloc[self.test_index]

    def rnaseq_features_as_numpy_array(self):
        '''
        returns numpy array of all features for all samples
        '''
        return self.df[self.feat_cols].values

    def export_label_encoder_mapping(self, outpath):
        '''
        writes the label encoder mapping
        '''
        classes = self.label_encoder.classes_
        encoding_df = pd.DataFrame({'tissue':classes, 
            'ID':list(range(len(classes)))})
        encoding_df.to_csv(outpath, sep='\t')

    def export_df(self, outpath):
        self.both_df.to_csv(outpath)

    def get_X_splits(self):
        '''
        return X matrices for the 3 splits
        '''
        train_X = self.both_df[self.both_df['subset']=='train'][self.feat_cols].values
        valid_X = self.both_df[self.both_df['subset']=='valid'][self.feat_cols].values
        test_X = self.both_df[self.both_df['subset']=='test'][self.feat_cols].values

        return train_X, valid_X, test_X

    def get_y_splits(self):
        '''
        return y matrices for the 3 splits
        '''
        train_y = self.both_df[self.both_df['subset']=='train']['tissue_class_int'].values
        valid_y = self.both_df[self.both_df['subset']=='valid']['tissue_class_int'].values
        test_y = self.both_df[self.both_df['subset']=='test']['tissue_class_int'].values

        return train_y, valid_y, test_y

    def get_sample_splits(self):
        '''
        return sample_names for the 3 splits
        '''
        train_s = self.both_df[self.both_df['subset']=='train']['sample_name'].values
        valid_s = self.both_df[self.both_df['subset']=='valid']['sample_name'].values
        test_s = self.both_df[self.both_df['subset']=='test']['sample_name'].values

        return train_s, valid_s, test_s

    def test_split_strat(self):
        '''
        verify that the stratified split is good
        '''
        print_counts(self.both_df['tissue_class_int'])
        print_counts(self.train_df['tissue_class_int'])
        print_counts(self.valid_df['tissue_class_int'])
        print_counts(self.test_df['tissue_class_int'])

        print("should only contain {'train', 'valid', 'test'}")
        print(set(self.both_df['subset'].values))

        print_counts(self.both_df[self.both_df['subset']=='train']['tissue_class_int'])
        print_counts(self.both_df[self.both_df['subset']=='valid']['tissue_class_int'])
        print_counts(self.both_df[self.both_df['subset']=='test']['tissue_class_int'])


        assert len(set(self.train_index).intersection(set(self.test_index)))==0
        assert len(set(self.train_index).intersection(set(self.valid_index)))==0
        assert len(set(self.valid_index).intersection(set(self.test_index)))==0

        assert len(self.both_df[self.both_df['subset']==''])==0

        assert len(self.test_index)+len(self.valid_index)+len(self.train_index)==len(self.both_df)
        print("passed tests")

def print_counts(ints):
    cats, counts = np.unique(ints, return_counts=True)
    cats_counts = list(zip(cats, counts))
    cats_counts.sort(key=lambda x: x[0])
    for cat, count in zip(cats, counts):
        print(cat, count) 

if __name__ == '__main__':
    args = parse_args()

    crd = CombinedRNASeqData(args.combined_rnaseq_data, 
                    args.combined_cl_metadata)

    Xs = crd.get_X_splits()
    ys = crd.get_y_splits()
    samples = crd.get_sample_splits()

    # save X
    for X, subset in zip(Xs, ['train', 'valid', 'test']):
        filename = '.'.join(['rnaseq_features', subset, 'npy'])
        np.save(os.path.join(args.output_dir, filename), X)

    # save labels
    for y, subset in zip(ys, ['train', 'valid', 'test']):
        filename = '.'.join(['rnaseq_features_label', subset, 'y.1'])
        np.savetxt(os.path.join(args.output_dir, filename), y)

    # save sample ids
    for sample, subset in zip(samples, ['train', 'valid', 'test']):
        filename = '.'.join(['rnaseq_features', subset, 'sample'])
        with open(os.path.join(args.output_dir, filename), 'w') as f:
            for s in sample:
                f.write(s+'\n')

    # export label encoder mapping
    crd.export_label_encoder_mapping(
        os.path.join(args.output_dir, 'label_map_vy1.txt')
    )

    # export full dataframe for reference
    crd.export_df(
        os.path.join(args.output_dir, 'rnaseq_data.csv')
    )