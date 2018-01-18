import numpy as np
import os
import argparse


MODEL_TYPES = {'rand': 0, 'static': 1, 'non-static': 2}

class CONFIG(object):

    def __init__(self, learning_rate, n_epochs,
                 batch_size, device, args):
        self.FILTER_SIZE_LIST = [3, 4, 5]
        self.FILTERS_PER_LAYER = [100, 100, 100]
        self.POOL_SIZE_LIST = [2, 2, 2]
        self.DROPOUT_LIST = [0.5, 0.5]
        self.LEARNING_RATE = learning_rate
        self.NUM_EPOCHS = n_epochs
        self.BATCH_SIZE = batch_size
        self.DEVICE = device
        self.MODEL_TYPE = args.model_type
        self.SEQUENCE_LEN = args.s_len
        self.EMBEDDING_DIM = args.dim
        self.NB_WORDS = args.nb_words
        self.CKPT_PATH = args.ckpt_path

        print('Reading Word Vectors...')
        self.EMBEDDING_INDEX, self.WORDS = create_embedding_index(args.e_src, self.EMBEDDING_DIM)
        self.WORD_COUNT = len(self.WORDS)
        print('Found {} words'.format(len(self.EMBEDDING_INDEX)))

    def is_use_embedding(self):
        return not self.MODEL_TYPE == 0

    def is_embedding_trainable(self):
        return not self.MODEL_TYPE == 1

    def add_embedding_matrix(self, word_index):
        self.NB_WORDS = min(self.NB_WORDS, len(word_index))
        
        self.EMBEDDING_MATRIX = np.zeros((self.NB_WORDS+1, self.EMBEDDING_DIM))
        for word, i in word_index.items():
            if i > self.NB_WORDS:
                continue
            embedding_vector = self.EMBEDDING_INDEX.get(word)

            if embedding_vector is not None:
                self.EMBEDDING_MATRIX[i] = embedding_vector
        self.WORDS = len(list(word_index.keys()))



def create_embedding_index(path, embedding_dim=100):

    embedding_index = {}
    words = []
    with open(path, encoding='utf-8') as matrix_file:
        for word_vector in matrix_file:
            word_vector = word_vector.split()
            words.append(word_vector[0])
            embedding_index[word_vector[0]] = np.asarray([word_vector[1:embedding_dim+1]],
                                                         dtype=np.float32)

    return embedding_index, words


def load_dataset(path, text_file_name='input.txt', label_file_name='label.txt'):

    texts = list(open(os.path.join(path, text_file_name), 'r').readlines())
    label_file = open(os.path.join(path, label_file_name), 'r')

    labels_to_int = {}
    current_id = 0
    label_ids = []
    for id, line in enumerate(label_file):
        label = str(line.strip())
        if label not in labels_to_int:
            labels_to_int[label] = current_id
            current_id += 1
        label_ids.append(labels_to_int[label])

    return texts, labels_to_int, len(labels_to_int.keys()), label_ids


def prepare_argparser():

    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, default='./datasets/',
                        help="Path to directory containing text file and label file dataset. "
                             "Default - './datasets/'")
    parser.add_argument('--e_src', type=str, default='./embeddings/e_vectors.txt',
                        help="File containing embedding vectors. "
                             "Default - './embeddings/e_vectors.txt'")
    parser.add_argument('--ckpt_path', type=str, default='./checkpoint/text_classifier.ckpt',
                        help="Path to directory where the checkpoint of the model should be stored."
                             " If the directory doesn't exist, it will be created. "
                             "Default - './checkpoint/'")
    parser.add_argument('--s_len', type=int, default=64,
                        help="Maximum length of the input sequence. Default - 64")
    parser.add_argument('--dim', type=int, default=100,
                        help="Dimensions of the embedding vector space to be utilized. Default - 100")
    parser.add_argument('--nb_words', type=int, default=10000,
                        help="Numbers of words to keep from the dataset.")
    parser.add_argument('--model_type', type=str, default='rand',
                        help="Name of the model." 
                             "Possible values - 'rand', 'static', 'non-static'"
                             "Check CNN sentence classifier paper here for details -> "
                             "https://arxiv.org/abs/1408.5882. Default - 'rand'")
    parser.add_argument('-h_file', type=str, default='tunning_params.txt',
                        help="In case of default model type a tunning_params.txt file is required."
                             " The file should contain hyperparameters in following order each on new line. "
                             "Learning Rate, Num Epochs, Mini-batch size, Device - CPU/GPU."
                             " Default - 0.001, 10, 32, GPU. Default 'tunning_params.txt'")
    return parser.parse_args()


def assert_and_compile_args(args):
    if not os.path.isdir(args.src):
        raise ValueError("Text file and label file directory - '{}' doesn't exist".format(args.src))
    if not os.path.exists(args.e_src):
        raise ValueError("File containing embedding vectors - '{}' doesn't exist".format(args.e_src))
    if args.model_type not in MODEL_TYPES:
        args.model_type = MODEL_TYPES['rand']
    else:
        args.model_type = MODEL_TYPES[args.model_type]

    if args.model_type == MODEL_TYPES['rand'] and os.path.exists(args.h_file):
        return _get_config(args, args.h_file), args
    else:
        return _get_config(args), args


def _get_config(args, path=None):
    if path == None:
        learning_rate = 1e-3
        n_epochs = 10
        batch_size = 32
        device = 'gpu'
    else:
        with open(path, 'r') as f:
            learning_rate = float(f.readline().strip())
            n_epochs = int(f.readline().strip())
            batch_size = int(f.readline().strip())
            device = str(f.readline().strip()).lower()

    return CONFIG(learning_rate, n_epochs, batch_size, device, args)
