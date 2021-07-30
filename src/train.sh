python tf_main.py --display_step 2 \
--valid_size 1024 \
--epochShuffle True \
--scale "" \
--cost "rms" \
--out_check_name "autoencoder" \
--outputFolder ../model \
--summaryFolder ../model \
--graph_type CenterLossAutoencoder \
--epoch_count 20 \
--batch_size 1000 \
--center_loss_lambdas 1 1 2 \
--learning_rate 0.0001 \
--center_loss_alpha 0.8 \
--trainx ../data/processed_ftp_data/rnaseq_features.train.npy \
--trainy ../data/processed_ftp_data/rnaseq_features_label.train.y.1 \
--valid ../data/processed_ftp_data/rnaseq_features.valid.npy \
--validy ../data/processed_ftp_data/rnaseq_features_label.valid.y.1

echo "Done"
