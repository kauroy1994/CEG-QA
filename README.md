# Repository for Causal Event Graph-Guided Language-based Spatiotemporal Question Answering Paper

This repository contains training code for baselines and methods implemented in the paper

## Library Installations
```
python -m pip install -r requirements.txt
```
## Baseline Methods
Tested with transE, distMult, CompIEx, and HolE embeddings

### Training Scripts for CLEVRER Dataset
```
python main.py --datapath data --datafile CLEVRER_data.pkl --baseline trans_e
python main.py --datapath data --datafile CLEVRER_data.pkl --baseline dm_e
python main.py --datapath data --datafile CLEVRER_data.pkl --baseline comp_e
python main.py --datapath data --datafile CLEVRER_data.pkl --baseline hyp_e
```

### Training Scripts for CLEVRER Humans Dataset
```
python main.py --datapath data --datafile CLEVRER_H_data.pkl --baseline trans_e
python main.py --datapath data --datafile CLEVRER_H_data.pkl --baseline dm_e
python main.py --datapath data --datafile CLEVRER_H_data.pkl --baseline comp_e
python main.py --datapath data --datafile CLEVRER_H_data.pkl --baseline hyp_e
```
## Proposed Method
Tested with both the CLEVRER and CLEVRER Humans Dataset

### Training Script for CLEVRER Dataset
```
python main.py --datapath data --datafile CLEVRER_data.pkl
```

### Training Scripts for CLEVRER Humans Dataset
```
python main.py --datapath data --datafile CLEVRER_H_data.pkl
```
