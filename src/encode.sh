python tf_main.py --display_step 2 \
--valid_size 1024 \
--epochShuffle True \
--scale "" \
--cost "rms" \
--outputFolder ../model \
--summaryFolder ../model \
--graph_type CenterLossAutoencoder \
--encode \
--out_check_name "autoencoder_ratchet" \
--center_loss_lambdas 1 1 2 \
--learning_rate 0.0001 \
--center_loss_alpha 0.8 \
--valid ../data/processed_ftp_data/rnaseq_features.valid.npy ../data/processed_ftp_data/rnaseq_features.test.npy ../data/processed_ftp_data/rnaseq_features.train.npy \
--validy ../data/processed_ftp_data/rnaseq_features_label.valid.y.1 ../data/processed_ftp_data/rnaseq_features_label.test.y.1 ../data/processed_ftp_data/rnaseq_features_label.train.y.1

echo "Done"