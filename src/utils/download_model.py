import os
from file_utils import get_file

response_collection_path="https://modac.cancer.gov/api/v2/dataObject/NCI_DOE_Archive/JDACS4C/JDACS4C_Pilot_1/rna_seq_latent_featurizer_using_center_loss_cost_function_CLRNA"
meta = "autoencoder_ratchet.meta"
index = "autoencoder_ratchet.index"
weights = "autoencoder_ratchet.data-00000-of-00001"

meta_url = response_collection_path  + "/" + meta 
index_url = response_collection_path  + "/" + index 
weights_url = response_collection_path  + "/" + weights

data_dest = "model" 

get_file(meta , meta_url, datadir=data_dest)
get_file(index , index_url, datadir=data_dest)
get_file(weights , weights_url, datadir=data_dest)
