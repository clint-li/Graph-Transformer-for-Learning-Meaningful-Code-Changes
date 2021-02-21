# Graph-Transformer-for-Learning-Meaningful-Code-Changes

Code for our FSE paper,

Graph Transformer for Graph-to-Sequence Learning. 

## 1. Environment Setup

The code is tested with Python 3.7. All dependencies are listed in [requirements.txt](requirement.txt).

### Installing C2NL

You may consider installing the C2NL package. C2NL requires Python 3.6 or higher. It also requires installing PyTorch version 1.3. CUDA is strongly recommended for speed, but not necessary. 

Run the following commands to clone the repository and install C2NL:

```
git clone https://github.com/wasiahmad/NeuralCodeSum.git
cd NeuralCodeSum; pip install -r requirements.txt; python setup.py develop
```

## 2. Datasets

The main graph dataset in our paper is given in the [dataset_BNCS](dataset_BNCS), the other four graph datasets are also given. Each graph dataset has six datasets. Each dataset is split in three folders:

- `train`: code transformations used for training the model (80% of the data);
- `dev`: code transformations used for validation of the model (10% of the data);
- `test`: code transformations used for testing the model (10% of the data);

Each folder contains two files:

- `buggy.txt` : the source code before the code transformation;
- `fixed.txt` : the source code after the code transformation;

The two pair of files represent a parallel corpus, where the i-th line in buggy.txt and the i-th line in fixed.txt reprensent the i-th transformation pair (*code_before*, *code_after*).

## 3. Train & TEST & Beam Search Test

Run the following commands to start train, test and beam search test. 
```
cd scripts/java
bash transformer_4.sh <GPU_NAME> <MODEL_NAME> <DATASET_NAME> <SAVE_PATH>
```

- `GPU_NAME`: The number of GPU;
- `MODEL_NAME`: the name of training model;
- `DATASET_NAME`: the name of using dataset;
- `SAVE_PATH`: the path of saved model;

###Example

```
cd scripts/java
bash transformer_4.sh 0 google_small google/50 tmp_google_small
```

In the end, we obtain the number of perfect predictions.

## Generated log files

While training and testing the models, a list of files sre generated inside the SAVE_PATH directory. The files are as follows.

- **MODEL_NAME.mdl**
  - Model file containing the parameters of the best model.
- **MODEL_NAME.mdl.checkpoint**
  - A model checkpoint, in case if we need to restart the training.
- **MODEL_NAME.txt**
  - Log file for training.
- **MODEL_NAME.json**
  - The predictions and gold references are dumped during validation.
- **MODEL_NAME_test.txt**
  - Log file for evaluation (greedy).
- **MODEL_NAME_test.json** 
  - The predictions and gold references are dumped during evaluation (greedy).
- **MODEL_NAME_test_predictions.txt**
  - The pure predictions are generated during evalution (greedy).
- **MODEL_NAME_buggy.txt**
  - The buggy code files for calculate perfect predictions.
- **MODEL_NAME_fixed.txt**
  - The fixed code files for calculate perfect predictions.
- **MODEL_NAME_beam.txt**
  - Log file for evaluation (beam).
- **MODEL_NAME_beam.json**
  - The predictions and gold references are dumped during evaluation (beam).
- **MODEL_NAME_beam_predictions.txt**
  - The pure predictions are generated during evaluation (beam).

## Citation

If you find the code useful, please cite our paper.
```

```




