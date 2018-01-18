import tensorflow as tf


__ACTIVATIONS = {
        'relu': tf.nn.relu,
        'softmax': tf.nn.softmax,
        'sigmoid': tf.nn.sigmoid,
        'tanh': tf.nn.tanh,
}


def __strides_for_tf(strides=(1, 1)):
    return [1, strides[0], strides[1], 1]


def __ksize_for_tf(k_size):
    return [1, k_size, k_size, 1]


def __apply_nonlin(tensor, act):
    return __ACTIVATIONS[act](tensor)


def __get_weights_and_bias(W_shape, b_shape, frozen=False):
    initializer=tf.contrib.layers.xavier_initializer()

    if not frozen:
        W = tf.Variable(initializer(W_shape))
        b = tf.Variable(tf.zeros(b_shape))
    else:
        W = tf.constant(initializer(W_shape))
        W = tf.constant(tf.zeros(b_shape))

    return W, b


def __weights_and_bias_from_args(weights, frozen=False):
    W, b = weights[0], weights[1]

    if not frozen:
        return tf.Variable(W), tf.Variable(b)
    return tf.constant(W), tf.constant(b)



def __conv1d_weights_given_tensor(tensor, k_size, n_k, frozen=False):
    m, n_W, n_C = tensor.get_shape().as_list()

    W_shape = [k_size, n_C, n_k]
    b_shape = [1, 1, n_k]

    return __get_weights_and_bias(W_shape, b_shape, frozen=frozen)


def __conv2d_weights_given_tensor(tensor, k_size, n_k, frozen=False):
    m, n_H, n_W, n_C = tensor.get_shape().as_list()

    W_shape = [k_size, k_size, n_C, n_k]
    b_shape = [1, 1, 1, n_k]

    return __get_weights_and_bias(W_shape, b_shape, frozen=frozen)


def __dense_weights_given_tensor(tensor, out, frozen=False):
    m, in_size = tensor.get_shape().as_list()

    W_shape = [in_size, out]
    b_shape = [1, out]

    return __get_weights_and_bias(W_shape, b_shape, frozen=frozen)


def concat(tensor_list, axis=-1):
    return tf.concat(tensor_list, axis=axis)


def dropout(tensor, keep_prob, name=None):
    if name:
        return tf.nn.dropout(tensor, keep_prob, name=name)
    else:
        return tf.nn.dropout(tensor, keep_prob)


def input(shape, dtype='float32', initial_vals=None, trainable=False, name=None):
    """
    Input to the computational graph.
    @params:
        shape - Shape of a single data point - list.
    @keyword args:
        initial_vals - If not None the the values of the input will be fixed to the given initial vals. (default False)
        trainable - If True the input will be trainable. Useful in case of style transfer. (default False)
        name  - Name of the input node. (default None - keep tensorflow names)
    @returns
        input_node - A tensorflow placeholder or constant or Variable depending upon values of 'initial_vals' and 
        'trainable' with batch dimension included.
    """
    assert isinstance(shape, list)

    if not trainable:
        shape = [None] + shape

        if name == None:
            return tf.placeholder(dtype=dtype, shape=shape)
        return tf.placeholder(dtype=dtype, shape=shape, name=name)

    if initial_vals is None:
        initializer = tf.contrib.layers.xavier_initializer()
        if name == None:
            return tf.Variable(initializer(shape))
        return tf.Variable(initializer(shape), name=name)

    if name == None:
        return tf.Variable(initial_vals)
    return tf.Variable(initial_vals, name=name)


def conv1d(tensor, k_size, n_k, strides=1,
           padding='SAME', nonlin=None, weights=None,
           frozen=False, name=None):
    """
    Apply 1D convolution to given tensor.
    @params:
        tensor  - 3D tensor with channels last.
        k_size  - Kernel size; an int
        n_k     - Number of output channels (number of kernels to be applied)
    @keyword args
        strides - Representing strides in 1D conv. (default 1)
        padding - Padding to the conv input (default 'SAME')
        nonlin  - Activation to be applied. Pass in either: 'relu', 'softmax', 'sigmoid', 'tanh', None (default None)
        weights - Initial values of weights and biases as tuple (weights, biases). (default None - to initialize using xavier initializer)
        frozen  - Set to True to make the layer untrainable. Eg. In case of transfer learning. (default False)
        name    - Name of the linear operation tensor. (default None - keep tensorflow names)
    @returns
        tensor  - Output of conv1d layer.
    
    Note: Keep nonlin None to get conv2d output w/o non-linearity.
    """
    assert nonlin == None or nonlin.lower() in list(__ACTIVATIONS.keys())

    if weights is not None:
        W, b = __weights_and_bias_from_args(weights, frozen)
    else:
        W, b = __conv1d_weights_given_tensor(tensor, k_size, n_k, frozen=frozen)


    if name == None:
        Z = tf.add(tf.nn.conv1d(tensor, W, padding=padding, stride=strides), b)
    else:
        Z = tf.add(tf.nn.conv1d(tensor, W, padding=padding, stride=strides), b, name=name)

    if nonlin != None:
        return __apply_nonlin(Z, nonlin.lower())
    return Z


def conv2d(tensor, k_size, n_k, strides=(1, 1),
           padding='SAME', nonlin=None, weights=None,
           frozen=False, name=None):
    """
    Apply 2D convolution to given tensor.
    @params:
        tensor  - 4D tensor with channels last.
        k_size  - Kernel size; an int
        n_k     - Number of output channels (number of kernels to be applied)
    @keyword args
        strides - Tuple representing strides in row, column respectively. (default (1, 1))
        padding - Padding to the conv input (default 'SAME')
        nonlin  - Activation to be applied. Pass in either: 'relu', 'softmax', 'sigmoid', 'tanh', None (default None)
        weights - Initial values of weights and biases as tuple (weights, biases). (default None - to initialize using xavier initializer)
        frozen  - Set to True to make the layer untrainable. Eg. In case of transfer learning. (default False)
        name    - Name of the linear operation tensor. (default None - keep tensorflow names)
    @returns
        tensor  - Output of conv2d layer.
    
    Note: Keep nonlin None to get conv2d output w/o non-linearity.
    """
    assert nonlin == None or nonlin.lower() in list(__ACTIVATIONS.keys())

    if weights is not None:
        W, b = __weights_and_bias_from_args(weights, frozen)
    else:
        W, b = __conv2d_weights_given_tensor(tensor, k_size, n_k, frozen=frozen)

    strides = __strides_for_tf(strides)

    if name == None:
        Z = tf.add(tf.nn.conv2d(tensor, W, padding=padding, strides=strides), b)
    else:
        Z = tf.add(tf.nn.conv2d(tensor, W, padding=padding, strides=strides), b, name=name)

    if nonlin != None:
        return __apply_nonlin(Z, nonlin.lower())
    return Z


def maxpool(tensor, k_size, strides=(1, 1),
            padding='VALID', name=None):
    """
    Maxx pooling layer.
    @params:
        tensor  - 4D tensor with channels last.
        k_size  - Kernel size; an int
    @keyword args
        strides - Tuple representing strides in row, column respectively. (default (1, 1))
        padding - Padding to the conv input (default 'SAME')
        name    - Name of the output tensor. (default None - keep tensorflow names)
    @returns
        tensor  - Output of max pooling layer.
    """
    k_size = __ksize_for_tf(k_size)
    strides = __strides_for_tf(strides)
    if len(tensor.get_shape().as_list()) == 3:
        tensor = tf.expand_dims(tensor, 3)

    assert len(tensor.get_shape().as_list()) == 4

    if name == None:
        return tf.nn.max_pool(tensor, ksize=k_size, strides=strides, padding=padding)

    return tf.nn.max_pool(tensor, ksize=k_size, strides=strides, padding=padding, name=name)


def flatten(tensor):
    """Flatten a 4D tensor to 2D"""
    m, n_H, n_W, n_C = tensor.get_shape().as_list()
    out = n_H * n_W * n_C
    return tf.reshape(tensor, shape=[-1, out])


def dense(tensor, out, nonlin=None, weights=None,
          frozen=False, name=None):
    """
    A fully connected layer.
    @params:
        tensor  - 4D tensor with channels last.
        out     - Number of output neurons; an int
    @keyword args
        nonlin  - Activation to be applied. Pass in either: 'relu', 'softmax', 'sigmoid', 'tanh', None (default None)
        weights - Initial values of weights and biases as tuple (weights, biases). (default None - to initialize using xavier initializer)
        frozen  - Set to True to make the layer untrainable. Eg. In case of transfer learning. (default False)
        name    - Name of the linear operation tensor. (default None - keep tensorflow names)
    @returns
        tensor  - Output of Dense layer. 
    """
    assert nonlin == None or nonlin.lower() in list(__ACTIVATIONS.keys())

    if weights is None:
        W, b = __dense_weights_given_tensor(tensor, out, frozen=frozen)
    else:
        W, b = __weights_and_bias_from_args(weights, frozen=frozen)

    if name == None:
        Z = tf.add(tf.matmul(tensor, W), b)
    else:
        Z = tf.add(tf.matmul(A, W), b, name=name)

    if nonlin != None:
        return __apply_nonlin(Z, nonlin.lower())
    return Z


def embeddings(tensor, input_dim, output_dim, weights=None, input_length=100, frozen=False):
    """
    An embedding layer for text to Tensor.
    @params:
        input_dim  - Vocabulary Size.
        output_dim - Dimension of the embedding
    @keyword args
        weights    - The predefined embedding matrix. Keep it None to make new word embeddings for vocabulary(default None)
        input_len  - The max size of incoming string. (default 100)
        frozen     - Set to True to make the layer untrainable. Eg. In case of transfer learning. (default False)
    @returns
        tensor     - A tensorflow tensor of shape [input_dim, output_dim]
    """

    m, len = tensor.get_shape().as_list()

    assert len == input_length

    if not frozen:
        if weights is None:
            word_embeddings = tf.get_variable("embeddings", shape=[input_dim, output_dim])
        else:
            word_embeddings = tf.Variable(weights, name="embeddings")
    else:
        assert weights is not None
        word_embeddings = tf.constant(weights, name="embeddings")

    embedded_word_ids = tf.nn.embedding_lookup(word_embeddings, tensor)
    return tf.cast(embedded_word_ids, tf.float32)
