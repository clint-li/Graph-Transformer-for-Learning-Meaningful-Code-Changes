# Graph-Transformer-for-Learning-Meaningful-Code-Changes

Code for our FSE paper,

Graph Transformer for Graph-to-Sequence Learning. 

## 1. Environment Setup

The code is tested with Python 3.7. All dependencies are listed in [requirements.txt](requirement.txt).

## 2. Datasets

The main graph dataset in our paper is given in the [dataset_BNCS](dataset_BNCS), the other four graph datasets are also given. Each graph dataset has six datasets. Each dataset is split in three folders:

- `train`: code transformations used for training the model (80% of the data);
- `eval`: code transformations used for validation of the model (10% of the data);
- `test`: code transformations used for testing the model (10% of the data);

Each folder contains two files:

- `buggy.txt` : the source code before the code transformation;
- `fixed.txt` : the source code after the code transformation;

The two pair of files represent a parallel corpus, where the i-th line in buggy.txt and the i-th line in fixed.txt reprensent the i-th transformation pair (*code_before*, *code_after*).

## 3. Train & TEST & Beam Search Test

```
cd script/java
bash transformer_4.sh <GPU_NAME> <MODEL_NAME> <DATASET_NAME> <SAVE_PATH>
- `GPU_NAME`: The 


