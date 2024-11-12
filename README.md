# DEEM

We provide the code detail files and related model files for the method DEEM mentioned in the paper “Discriminative Feature Decoupling Enhancement for Speech Forgery Detection”.

## Intro
- The purpose of this code project file is to build the DEEM algorithm needed for forged voice detection. The core idea of this algorithm is to decouple the foreground and background related features through feature decoupling, and to strengthen the decoupled features through graph attention mechanism, in order to achieve stronger and broader forged voice detection effects.

## Installation 



`requirements.txt` must be installed for execution. 

- Installing dependencies

```
pip install -r requirements.txt
```



- ASVspoof2019 dataset:

  https://datashare.ed.ac.uk/handle/10283/3336

  1. Download `LA.zip` and unzip it
  2. Set your dataset directory 

- FoR dataset：

  https://www.eecs.yorku.ca/~bil/Datasets/for-norm.tar.gz

  1. Download `for-norm.tar.gz` and decompress it
  2. Set your dataset directory 

  

## Instructions for Use
- The `data_utils.py` is used to process data input.It's a utility class.
- The `AE_decoupling_DEEM.py`  defined the autoencoder architecture for feature decoupling training.
- The `trainer2decoupling.py` use the synthetic dataset together with the forgery detection dataset to complete the decoupled training of the decoupled autoencoder.
- The `classifier_DEEM.py`uses the trained decoupled autoencoder to couple into the backend based on the heterogeneous graph attention mechanism, and uses it to realize the speech forgery detection task.
- The `valid_DEEM.py`  can use the already trained model to verify its performance.
- The `conf.yaml` defines the configuration parameters required by each model.
- The `eval_metrics.py` is a metric calculation utility class used to measure the performance of the algorithm.
- The `eer2.py` is also a calculation utility class for metrics used to measure the performance of an algorithm.
- The `basenet.py` provides some basic network models for reuse.
- 

To train decoupling module:

```
python trainer2decoupling.py
```

To train detection module:

```
python classifier_DEEM.py
```

## Pre-trained models

We provide pre-trained models for different datasets  in the 'model' folder.

- Decoupled feature-based classifier model for dataset 'FoR': `FoR.pth`
- Decoupled feature-based classifier model for dataset 'ASVSpoof2019LA': `ASV.pth`

