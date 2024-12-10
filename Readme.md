<center>
  <h1>Readme</h1>
</center>


This experiment is **Chinese-English Machine Translation**.

The task is to use the $\text{Seq2Seq}$ model with the $\text{Attention}$ mechanism to translate Chinese sentences into English sentences.



## Install

This experiment supports training and prediction of models on a GPU. If you wish to use a GPU for task execution, please ensure that the correct GPU drivers, as well as the corresponding versions of CUDA and cudnn, are installed.

The third-party libraries required for this project and their version information are stored in the `requirements.txt` file.

To install the required libraries in a new environment, use the following command in the terminal:

```bash
pip install -r requirements.txt
```

This will install the corresponding third-party libraries when setting up the project.



## Usage

The project code consists of three py files:

- `datapreprocessing`：Data preprocessing, which includes
  - Data cleaning
  - Tokenization
  - Chinese-English dictionary construction
  - Text encoding
  - Data loader construction
- `model`：Implementation of the network model architecture




  - Seq2Seq model: Encoder and Decoder built based on GRU
  - Manual implementation of the Attention mechanism (including three different alignment functions)
  - Two different training strategies (Teacher Forcing and {Free Running)
  - Two different decoding strategies (Greedy Search and Beam Search）
- `FER`：Model training, validation, testing processes, and BLEU evaluation metric



The dataset used for this experiment is located in `./data_short/nmt/en-cn`

- Training set: `train.txt`
- Validation set: `dev.txt`
- Test set: `test.txt`



**How to Run the Code**

```bash
python3 main.py
```



## Documents instruction

`./Image/FR_50.png`： A plot showing the training and validation loss curves for each iteration using the Free Running strategy.

`./Image/TF_50.png`：A plot showing the training and validation loss curves for each iteration using the Teacher Forcing strategy.

`./Image/attention_1.png`：The Attention heatmap for Example 1, which is not normalized.

`./Image/attention_1_scaling`：The normalized Attention heatmap for Example 1.

`./Image/attention_2_scaling.png`：The normalized Attention heatmap for Example 2.
