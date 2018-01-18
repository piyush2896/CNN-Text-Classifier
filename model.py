import tensorflow as tf
from nn.layers import maxpool, dense, embeddings, input
from nn.layers import flatten, conv1d, dropout, concat
from nn.model import Model


def load_model(config, n_classes=2):
    model = Model()
    X = model.add(input([config.SEQUENCE_LEN], dtype='int32', name="input"))

    if config.is_use_embedding():
        embedding = model.add(embeddings(X, config.WORD_COUNT, 
                                         config.EMBEDDING_DIM,
                                         weights=config.EMBEDDING_MATRIX,
                                         input_length=config.SEQUENCE_LEN,
                                         frozen=config.is_embedding_trainable()))
    else:
        embedding = model.add(embeddings(X, config.WORD_COUNT, 
                                         config.EMBEDDING_DIM,
                                         input_length=config.SEQUENCE_LEN,
                                         frozen=config.is_embedding_trainable()))

    dropout_1 = model.add(dropout(embedding, config.DROPOUT_LIST[0]))

    conv_list = []
    for k_size, n_C, k_pool in zip(config.FILTER_SIZE_LIST, config.FILTERS_PER_LAYER, config.POOL_SIZE_LIST):
        c = conv1d(dropout_1, k_size, n_C, nonlin='relu')
        p = maxpool(c, k_pool)
        conv_list.append(flatten(p))

    if len(conv_list) > 1:
        conv_out = model.add(concat(conv_list))
    else:
        conv_out = model.add(conv_list[0])

    dense_1 = model.add(dense(conv_out, 150, nonlin='relu'))
    dropout_2 = model.add(dropout(dense_1, config.DROPOUT_LIST[1]))
    out = model.add(dense(dropout_2, n_classes, nonlin='softmax'))
    model.compile(optimizer='rmsprop', loss='softmax_entropy',
                  learning_rate=config.LEARNING_RATE, ckpt_file=config.CKPT_PATH,
                  device=config.DEVICE)

    return model
