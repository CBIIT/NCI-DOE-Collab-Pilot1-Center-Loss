## RNA-seq latent featurizer using center loss cost function (CLRNA) v0.1
##### Authors: Stewart He (he6@llnl.gov), Jonathan E. Allen, Ya Ju Fan
##### Released: May 27th, 2021

### Description:
`CLRNA` is a python package built on Tensoflow to learn new features
for RNASeq data.

Predictive modeling of patient tumor growth response to drug treatment 
is severely limited by a lack of experimental data for training. 
Combining experimental data from several studies is an attractive approach, 
but presents two problems: batch effects and limited data for individual 
drugs and cell lines. Batch effects are caused by systematic 
procedural differences among studies, which causes systematic 
experimental outcome differences. Directly using these experimental 
results as features for machine learning commonly causes problems when 
training on one study and testing on another. This severely limits a 
model’s ability to perform well on new experiments. Even after 
combining studies, predicting outcomes on new patient tumors remains an 
open challenge.

We propose a semi-supervised, autoencoder-based, machine learning 
procedure, which learns a smaller set of gene expression features that 
are robust to batch effects using background information on a cell line 
or tissue’s tumor type. We implemented this reduced feature representation 
and show that the new feature space clusters strongly according to tumor 
type. This experiment is carried out across multiple studies: CCLE, CTRP, 
gCSI, GDSC, NCI60, and patient derived tumors. We hypothesize that using 
a batch effect resistant feature set across studies will improve 
prediction performance.

Genome Data Commons (GDC) gene expression profiles for publicly available 
human tissues and cell lines from NCI60 and CCLE were processed using 
the semi-supervised learning procedure. Our autoencoder repurposes the 
‘center loss’ (CL) cost function of 
[Wen et. al.](https://link.springer.com/chapter/10.1007/978-3-319-46478-7_31) 
to learn a more generalized set of features using the cell line or 
tissue’s tumor type. Classification is performed by branching network 
the ‘pinch’ layer of the autoencoder. The ‘pinch’ layer now gets fed into 
a classification layer as well as the decoder portion of the autoencoder.

The new cost function balances the reconstruction performance, with 
the classification and ‘center loss’ performance. Reconstruction 
performance ensures that the ‘pinch’ layer retains information about 
original gene expression while classification performance shapes the space 
so tumors of the same type of close together regardless of the source 
study. Using the ‘pinch’ layer as new features reduces the number of 
features from 17,000 genes to approximately 1000 features or as few as 
20 features.

The performance of this method is compared with traditional batch 
correction methods (e.g. 
[ComBat](https://academic.oup.com/biostatistics/article/8/1/118/252073?login=true)). 
Before applying these methods, individual samples clustered more strongly 
along study, a property that is not useful in many machine 
learning applications. We compare the new features from our ‘center 
loss’ autoencoder and ComBat using Silhouette score, the Calinski – 
Harabasz index, and the Davies – Bouldin index. All metrics show that using 
the prosed ‘center loss’ autoencoder features provide a latent space 
with better clusters than applying ComBat.

### Environment
```
> cd code/
> pip install -r requirements.txt
```

### Data
RNAseq data can be obtained from the Pilot 1 FTP site 
[here](https://ftp.mcs.anl.gov/pub/candle/public/benchmarks/Pilot1/). 
Follow these instructions to process data into something that CLRNA can process.

#### Step 1:
Download RNAseq expression and cell line data 
[here](https://ftp.mcs.anl.gov/pub/candle/public/benchmarks/Pilot1/combo/combined_rnaseq_data) or
use the following commands.
```
> cd data/ftp_data/ 
> bash wget_ftp_data.sh 
--2021-07-28 14:39:17--  https://ftp.mcs.anl.gov/pub/candle/public/benchmarks/Pilot1/combo/combined_rnaseq_data
Resolving ftp.mcs.anl.gov (ftp.mcs.anl.gov)... 140.221.6.23
Connecting to ftp.mcs.anl.gov (ftp.mcs.anl.gov)|140.221.6.23|:443... connected.
HTTP request sent, awaiting response... 200 OK
Length: 1349730908 (1.3G) [text/plain]
Saving to: 'combined_rnaseq_data.2'

100%[=========================================>] 1,349,730,908 2.41MB/s   in 11m 51s

2021-07-28 14:51:08 (1.81 MB/s) - 'combined_rnaseq_data' saved [1349730908/1349730908]

--2021-07-28 14:51:08--  https://ftp.mcs.anl.gov/pub/candle/public/benchmarks/Pilot1/combo/combined_cl_metadata
Resolving ftp.mcs.anl.gov (ftp.mcs.anl.gov)... 140.221.6.23
Connecting to ftp.mcs.anl.gov (ftp.mcs.anl.gov)|140.221.6.23|:443... connected.
HTTP request sent, awaiting response... 200 OK
Length: 2683476 (2.6M) [text/plain]
Saving to: 'combined_cl_metadata.1'

100%[==========================================>] 2,683,476   1.92MB/s   in 1.3s

2021-07-28 14:51:10 (1.92 MB/s) - 'combined_cl_metadata' saved [2683476/2683476]
> ls
combined_cl_metadata  combined_rnaseq_data  wget_ftp_data.sh
```

#### Step 2:
Preprocess data into suitable format.
```
> cd code/ 
> python preprocess_ftp_data.py 
num rnaseq samples 15196
num cell line labels 15196
num after merge 15196
number of classes with more than 100 examples: 31
number of classes with fewer than 100 examples: 37
dropped classes with fewer than 100 examples
14373
expecting around 12642
```

### Training and Encoding
#### Step 1:
```
> bash train.sh
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
> bash encode.sh
...
checkpoint is ../model/autoencoder_ratchet
restoring from previous run
dumping to  ../model/rnaseq_features_label.valid.y.encoded
dumping to  ../model/rnaseq_features_label.test.y.encoded
dumping to  ../model/rnaseq_features_label.train.y.encoded
Done
> bash visualize.sh
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
### Step 2: Output values
Encoded features will be saved to ```/model/```. It will have the same
filename as the input except '.encoded' will be appended to the end of
the filename e.g., ```rnaseq_features_label.test.y.encoded.npy```. These features
handle batch effect which can be seen in the plots produced 
e.g.,```rnaseq_features_label.test.y.encoded.joined.png```. Example plots
can be found in ```/code/rnaseq_features_label.*.y.encoded.joined.png```

### Explanation 1: Train Center Loss Autoencoder
>python tf_main.py \\\
>--display_step 2 \\\
>--valid_size 1024 \\\
>--epochShuffle True \\\
>--scale "" \\\
>--cost "rms" \\\
>--out_check_name "autoencoder" \\\
>--outputFolder <output_folder> \\\
>--summaryFolder <summary_folder> \\\
>--graph_type CenterLossAutoencoder \\\
>--epoch_count 100 \\\
>--batch_size 1000 \\\
>--center_loss_lambdas 1 1 2 \\\
>--learning_rate 0.0001 \\\
>--center_loss_alpha 0.8 \\\
>--trainx <train_X> \\\
>--trainy <train_y> \\\
>--valid <valid_X> \\\
>--validy <valid_y>

### Explanation 2: Encode features
>python tf_main.py \\\
>--display_step 2 \\\
>--valid_size 1024 \\\
>--epochShuffle True \\\
>--scale "" \\\
>--cost "rms" \\\
>--out_check_name "autoencoder" \\\
>--outputFolder /g/g19/he6/lustre2/pilot_models/release_runs/refactor \\\
>--summaryFolder /g/g19/he6/lustre2/pilot_models/release_runs/refactor \\\
>--graph_type CenterLossAutoencoder \\\
>--encode \\\
>--out_check_name "autoencoder_ratchet" \\\
>--center_loss_lambdas 1 1 2 \\\
>--learning_rate 0.0001 \\\
>--center_loss_alpha 0.8 \\\
>--valid <encode_X> \\\
>--validy <encode_y>

### Glossary:
**output_folder:** Output directory. Model checkpoints will be saved here.

**summary_folder:** Usually the same as output_directory. 
    Model training summary will be saved here.
    
**train_X/valid_X:** Path to RNA seq features saved as a numpy file. One row
    per sample 17743 columns per sample. This file is generated in step 1 of the data section.
    
**train_y/valid_y:** Path to tumor features saved as a txt file. One row per
    sample. Each class should be represented as an integer. e.g 1, 2, 3. This file is generated in step 2 of the data section.

**encode_X/encode_y:** Files to be encoded. Follows the same format as
    training and validation files. Encode_y can be made up classifications.

### License
This is distributed under the terms of the MIT license.

All new contributions must be made under both the MIT licenses.

See LICENSE, COPYRIGHT, and NOTICE for details.

LLNL-CODE-824233
