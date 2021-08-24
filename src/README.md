## RNA-Seq Latent Featurizer Using Center Loss Cost Function (CLRNA) v0.1
##### Authors: Stewart He (he6@llnl.gov), Jonathan E. Allen, Ya Ju Fan
##### Released: May 27th, 2021

### Description
`CLRNA` is a Python package built on Tensorflow to learn new features for RNA-Seq data.

Predictive modeling of patient tumor growth response to drug treatment is severely limited by a lack of experimental data for training.  Combining experimental data from several studies is an attractive approach, but presents two problems: batch effects and limited data for individual drugs and cell lines. Batch effects are caused by systematic procedural differences among studies, which causes systematic experimental outcome differences. Directly using these experimental results as features for machine learning commonly causes problems when training on one study and testing on another. This severely limits a model’s ability to perform well on new experiments. Even after combining studies, predicting outcomes on new patient tumors remains an open challenge.

This semi-supervised, autoencoder-based, machine learning procedure learns a smaller set of gene expression features that are resistant to batch effects using background information on a cell line or tissue’s tumor type. The authors of this model implemented this reduced feature representation and show that the new feature space clusters strongly according to tumor type. The authors carried out experiments across multiple studies: Cancer Cell Line Encyclopedia ([CCLE](https://sites.broadinstitute.org/ccle/)), Cancer Therapeutics Response Portal ([CTRP](https://portals.broadinstitute.org/ctrp.v2.1/)), the Genentech Cell Line Screening Initiative ([gCSI](https://pharmacodb.pmgenomics.ca/datasets/4)), Genomics of Drug Sensitivity in Cancer ([GDSC](https://www.cancerrxgene.org/)), [NCI-60](https://discover.nci.nih.gov/cellminer/home.do), and patient derived tumors. The system downloads the data in this example from the [Cancer Drug Response Prediction Dataset](https://modac.cancer.gov/searchTab?dme_data_id=NCI-DME-MS01-8088592) in the Model and Data Clearinghouse (MoDaC). This method produces features that are resistant to batch effects.

The authors processed Genomic Data Commons (GDC) gene expression profiles for publicly available human tissues. The authors processed cell lines from NCI-60 and CCLE using the semi-supervised learning procedure. The autoencoder repurposes the ‘center loss’ cost function of [Wen et. al.](https://link.springer.com/chapter/10.1007/978-3-319-46478-7_31) to learn a more generalized set of features using the cell line or tissue’s tumor type. The authors performed classification by branching the network at the ‘pinch’ layer of the autoencoder. The authors fed activations from the 'pinch' layer forward to the decoder and the classification network. 

The new cost function is a weighted combination of three terms: reconstruction performance, classification performance, and ‘center loss’ performance. Reconstruction performance ensures that the ‘pinch’ layer retains information about original gene expression while classification performance shapes the space so tumors of the same type are close together regardless of the source study. Originally, representing each tumor required 17,000 gene activation features. However, with the 'pinch' layer, representing each tumor requires only 1,000 features or, with some loss in predictive performance, as few as 20 features.
 
The authors compared performance of this method with traditional batch correction methods (such as  [ComBat](https://academic.oup.com/biostatistics/article/8/1/118/252073?login=true)).  Before applying these methods, individual samples clustered more strongly along study lines, a property that is not useful in many machine learning applications. The authors compared the new features from the ‘center loss’ autoencoder and ComBat using Silhouette score, the Calinski-Harabasz index, and the Davies-Bouldin index. All metrics show that the ‘center loss’ autoencoder features provide a latent space with better clusters than applying ComBat.

### Setup
To set up the Python environment needed to run the system:
1. Install [conda](https://docs.conda.io/en/latest/) package manager.
2. Clone this repository.
3. Create the environment as shown below.

```bash
   conda env create -f environment.yml -n center_loss 
   conda activate center_loss 
```

### Data Download
Download data and convert it to tabular data that the system uses to train the CLRNA process:
1. Create an account on [MoDaC](https://modac.cancer.gov). 
2. Run the script [./src/utils/download_data.py](./utils/download_data.py). This script downloads from MoDaC the following data: 
   * RNA-Seq expressions
   * combined_cl_metadata
3. When prompted by the training and test scripts, enter your MoDaC credentials.


#### Data Preprocessing 
To preprocess the downloaded data into suitable format for CLRNA, run the following commands:
```
$ cd ./src/ 
$ python preprocess_ftp_data.py 
```
Here is example output from running these commands:
```
num rnaseq samples 15196
num cell line labels 15196
num after merge 15196
number of classes with more than 100 examples: 31
number of classes with fewer than 100 examples: 37
dropped classes with fewer than 100 examples
14373
expecting around 12642
```

### Training the Model  
To train the default model, run the following commands:
```
$ cd ./src
$ bash train.sh
```
Here is example output from running these commands:
```
...
2021-07-28 15:29
epoch 13
epoch 14
new best score 0.5080255
2021-07-28 15:31
epoch 15
epoch 16
new best score 0.47243702
2021-07-28 15:32
epoch 17
epoch 18
new best score 0.45152712
2021-07-28 15:34
epoch 19
Optimization Finished!
Done
```


### Download a Trained Model
To download a trained model instead of training a model, follow these steps:
1. Create an account on ([MoDaC](https://modac.cancer.gov). 
2. Run the script [./src/utils/download_model.py](./utils/download_model.py). 
3. When prompted by the training and test scripts, enter your MoDaC credentials.

### Encoding the Samples 
To use trained model to encode the RNA-Seq samples, run the following commands:
```
$ cd ./src/ 
$ bash encode.sh
```
Here is example output from running these commands:
```
...
checkpoint is ../model/autoencoder_ratchet
restoring from previous run
dumping to  ../model/rnaseq_features_label.valid.y.encoded
dumping to  ../model/rnaseq_features_label.test.y.encoded
dumping to  ../model/rnaseq_features_label.train.y.encoded
Done
```

The system saves encoded features to ```./model/```. The system saves the features with the same filename as the input except the system appends '.encoded' to the end of the filename (such as ```rnaseq_features_label.test.y.encoded.npy```).

### Visualization of Results 
The plots below show that this method clusters tumors together in latent space despite the presence of the batch effect in the input RNASeq data. Run this command:
```
$ bash visualize.sh
```
Here is example output from running this command:
```
explained variance
[0.2411185  0.2173541  0.11214621]
../data/processed_ftp_data/rnaseq_features_label.test.y.1
('#labels', (1438,), '#transformed', (1438, 3))
explained variance
[0.24758257 0.20721595 0.11479136]
../data/processed_ftp_data/rnaseq_features_label.train.y.1
('#labels', (2000,), '#transformed', (2000, 3))
explained variance
[0.24328862 0.21127205 0.11071263]
../data/processed_ftp_data/rnaseq_features_label.valid.y.1
('#labels', (1437,), '#transformed', (1437, 3))
Done
```
The [../figures](../figures) folder contains example plots: 
```rnaseq_features_label.*.y.encoded.joined.png```

Here is an example of the encoded samples using Principal Component Analysis and 3D [TSNE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html) plots with three different [perplexity](https://scikit-learn.org/stable/auto_examples/manifold/plot_t_sne_perplexity.html) values.

<img src="../figures/rnaseq_features_label.test.y.encoded.joined.png" alt="drawing" width="1200"/> 


### Glossary
**output_folder:** Output directory. The system saves model checkpoints here.

**summary_folder:** Usually the same as output_directory. The system saves the model training summary here.
    
**train_X/valid_X:** Path to RNA-Seq features saved as a numpy file. One row per sample and 17,743 columns per sample. The preprocess_ftp_data.py script generates these files.

**train_y/valid_y:** Path to tumor features saved as a text file. One row per sample. Represent each class as an integer, such as 1, 2, 3. The preprocess_ftp_data.py script generates these files.

**encode_X/encode_y:** Input files. Using the 'pinch' layer latent space encodes these files. These files must follow the same format as training and validation files. Encode_y can be fake classifications.

### License
The authors distributed this under the terms of the [MIT license](../LICENSE).

All new contributions must be made under the [MIT license](../LICENSE).

For details, refer to [LICENSE](../LICENSE), [COPYRIGHT](../COPYRIGHT), and [NOTICE](../NOTICE).

LLNL-CODE-824233
