# SOLO in Biomedical Imaging

Please refer to the codes section in the last part of report for more details.

## Files

*   `data_loader.py`:  the data loader to load data from the given directionary
*   `architecture.py`:  the classes of FPNs (ResNet and GRU) and SOLO head
*   `config.py`: the hyperparameter candidates for SOLO
*   `solo_train.py` and `solo_rnn_training.py`: the training pipeline for SOLO (and with GRU)
*   `map.py`: the functions of computing mAP scores w.r.t. full/25/50 mask size
*   `solo_visulization.ipynb`: the demonstration of the results

## Implementation

Put the above files under one directionary and run `solo_train.py` or `solo_rnn_train.py`. 
