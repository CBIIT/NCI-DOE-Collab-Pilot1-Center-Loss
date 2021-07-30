
python clusterPlot.py ../model/rnaseq_features_label.test.y.encoded.npy ../data/processed_ftp_data/rnaseq_features_label.test.y.1 .cluster.png 
python clusterPlot.py ../model/rnaseq_features_label.train.y.encoded.npy ../data/processed_ftp_data/rnaseq_features_label.train.y.1 .cluster.png 
python clusterPlot.py ../model/rnaseq_features_label.valid.y.encoded.npy ../data/processed_ftp_data/rnaseq_features_label.valid.y.1 .cluster.png 

echo "Done"