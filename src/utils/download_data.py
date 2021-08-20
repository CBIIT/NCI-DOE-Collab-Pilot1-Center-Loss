import os
from file_utils import get_file

response_collection_path="https://modac.cancer.gov/api/v2/dataObject/NCI_DOE_Archive/JDACS4C/JDACS4C_Pilot_1/cancer_drug_response_prediction_dataset"
combined_cl = "combined_cl_metadata"
cl_url = response_collection_path  + "/" + combined_cl 

rnaseq = "combined_rnaseq_data"
rnaseq_url = response_collection_path + "/" + rnaseq
data_dest = os.path.join("data", "ftp_data")

get_file(combined_cl , cl_url, datadir=data_dest)
get_file(rnaseq, rnaseq_url, datadir=data_dest)
