# CNN Text Classifier
This application is built by replicating work done in this paper [here](https://arxiv.org/abs/1408.5882).

## Dependencies
1. Tensorflow
2. Keras (Well, I wanted to avoid it entirely but had to use it for some data preprocessing :P)
3. Matplotlib
4. NumPy

## Usage

```
usage: train.py [-h] [--src SRC] [--e_src E_SRC] [--ckpt_path CKPT_PATH]
                [--s_len S_LEN] [--dim DIM] [--nb_words NB_WORDS]
                [--model_type MODEL_TYPE] [-h_file H_FILE]

optional arguments:
  -h, --help            show this help message and exit
  --src SRC             Path to directory containing text file and label file
                        dataset. Default - './datasets/'
  --e_src E_SRC         File containing embedding vectors. Default -
                        './embeddings/e_vectors.txt'
  --ckpt_path CKPT_PATH
                        Path to directory where the checkpoint of the model
                        should be stored. If the directory doesn't exist, it
                        will be created. Default - './checkpoint/'
  --s_len S_LEN         Maximum length of the input sequence. Default - 64
  --dim DIM             Dimensions of the embedding vector space to be
                        utilized. Default - 100
  --nb_words NB_WORDS   Numbers of words to keep from the dataset.
  --model_type MODEL_TYPE
                        Name of the model.Possible values - 'rand', 'static',
                        'non-static'Check CNN sentence classifier paper here
                        for details -> https://arxiv.org/abs/1408.5882.
                        Default - 'rand'
  -h_file H_FILE        In case of default model type a tunning_params.txt
                        file is required. The file should contain
                        hyperparameters in following order each on new line.
                        Learning Rate, Num Epochs, Mini-batch size, Device -
                        CPU/GPU. Default - 0.001, 10, 32, GPU.
```

## Model
![](images\model.png)
