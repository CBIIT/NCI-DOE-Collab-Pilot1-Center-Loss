# NCI-DOE-Collab-Pilot1-Semi-Supervised-Feature-Learning-with-Center-Loss

### Description
This software proposes a semi-supervised, autoencoder-based, machine learning procedure, which learns a smaller set of gene expression features that are robust to batch effects using background information on a cell line or tissue’s tumor type. We implemented this reduced feature representation and show that the new feature space clusters strongly according to tumor type. This experiment is carried out across multiple studies: CCLE, CTRP, gCSI, GDSC,NCI60, and patient derived tumors. We hypothesize that using a batch effect resistant feature set across studies will improve prediction performance.

### User Community
Researchers interested in the following topics:
* Primary: Cancer biology data modeling
* Secondary: Machine Learning; bioinformatics; computational biology

### Usability
The current code can be used by a data scientist experienced in Python and the domain.

### Uniqueness
The new cost function balances the reconstruction performance, with the classification and ‘center loss’ performance. Reconstruction performance ensures that the ‘pinch’ layer retains information about original gene expression while classification performance shapes the space so tumors of the same type of close together regardless of the source study. Using the ‘pinch’ layer as new features reduces the number of features from 17,000 genes to approximately 1000 features or as few as 20 features. We compare the new features from our ‘center loss’ autoencoder and ComBat using Silhouette score, the Calinski – Harabaszindex, and the Davies – Bouldin index. All metrics show that using the proposed ‘center loss’ autoencoder features provide a latent space with better clusters than applying ComBat.

### Components
This capability provides the following components:

* Scripts to download and process RNAseq expression and cell line data. 
* Script to train the autoencoder model

* The trained model

* Scripts to encode the RNAseq expression and visualize the reduced dimension resutls 

### Publication
Partin, A., Brettin, T., Evrard, Y.A. et al. Learning curves for drug response prediction in cancer cell lines. BMC Bioinformatics 22, 252 (2021). [https://doi.org/10.1186/s12859-021-04163-y](https://doi.org/10.1186/s12859-021-04163-y)


### Technical Details
Refer to this [README](./src/README.md).
