# NCI-DOE Collaboration Pilot 1: Semi-supervised Feature Learning with Center Loss

### Description
This software proposes a semi-supervised, autoencoder-based, machine learning procedure. This procedure learns a smaller set of gene expression features that are robust to batch effects using background information on a cell line or tissue’s tumor type. The authors of this model implemented this reduced feature representation and show that the new feature space clusters strongly according to tumor type. This experiment is carried out across multiple studies: CCLE, CTRP, gCSI, GDSC, NCI60, and patient derived tumors. The authors of this model hypothesize that using a batch effect resistant feature set across studies will improve prediction performance.

### User Community
Researchers interested in the following topics:
* Primary: Cancer biology data modeling
* Secondary: Machine learning; bioinformatics; computational biology

### Usability
The current code can be used by a data scientist experienced in Python and the domain.

### Uniqueness
The new cost function balances the reconstruction performance, with the classification and ‘center loss’ performance. Reconstruction performance ensures that the ‘pinch’ layer retains information about original gene expression while classification performance shapes the space so tumors of the same type are close together regardless of the source study. Using the ‘pinch’ layer as new features reduces the number of features from 17,000 genes to approximately 1,000 features or as few as 20 features. The authors of this model compare the new features from the ‘center loss’ autoencoder and ComBat using Silhouette score, the Calinski-Harabaszindex, and the Davies-Bouldin index. All metrics show that the ‘center loss’ autoencoder features provide a latent space with better clusters than applying ComBat.

### Components
This capability provides the following components:
* Scripts in this repository: 
 * Scripts to download and process RNA-Seq expression and cell line data. 
 * Script to train the autoencoder model
 * Scripts to encode the RNA-Seq expression and visualize the reduced dimension results 
* The trained model in the [CLRNA](https://modac.cancer.gov/searchTab?dme_data_id=NCI-DME-MS01-9815585) asset in the Model and Data Clearinghouse (MoDaC).

### Technical Details
Refer to this [README](./src/README.md).
