from util import *
from model import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from visualize import plot_stats


def run():
    args = prepare_argparser()
    config, args = assert_and_compile_args(args)

    print('Loading Dataset')
    texts, labels_to_int, n_classes, labels = load_dataset(args.src)
    print('Found {} lines and {} labels'.format(len(texts), n_classes))

    tokenizer = Tokenizer(num_words=config.NB_WORDS)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    word_index = tokenizer.word_index
    print('Found {} unique tokens.'.format(len(word_index)))

    data = pad_sequences(sequences, maxlen=config.SEQUENCE_LEN)

    # Transform labels to be categorical variables
    labels = np.array(to_categorical(np.asarray(labels)), dtype=np.float32)
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)

    print('Compiling Embedding Matrix')
    config.add_embedding_matrix(word_index)

    print('Loading Model')
    model = load_model(config, n_classes)

    print('Start Training...')
    hist = model.fit((data, labels), validation_split=0.1,
                     epochs=config.NUM_EPOCHS, batch_size=config.BATCH_SIZE)
    plot_stats(hist)

if __name__ == '__main__':
    run()
