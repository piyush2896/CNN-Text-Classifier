import tensorflow as tf
from nn.losses import *
from nn.metrics import *
from nn.preprocessor import *
from tqdm import tqdm
import shutil
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

OPTIMIZERS = {
        'adam': tf.train.AdamOptimizer,
        'rmsprop': tf.train.RMSPropOptimizer,
        'SGD': tf.train.GradientDescentOptimizer,
        'adagrad': tf.train.AdagradOptimizer
}

LOSSES = {
        'binary_entropy': binary_entropy,
        'softmax_entropy': softmax_entropy,
        'mse': mse
}

METRICS = {
        'binary_entropy': binary_accuracy,
        'softmax_entropy': multiclass_accuracy,
        'mse': explained_variance
}


class Model(object):

    def __init__(self):
        self.layers = []

    def __convert_to_one_hot(self, preds):
        m = preds.shape[0]
        labels = np.argmax(preds, axis=1)
        for i in range(m):
            preds[labels == i, i] = 1
            preds[labels != i, i] = 1
        return preds

    def __make_label_placeholder(self):
        out = self.layers[-1]
        m, n_out = out.get_shape().as_list()
        self.Y = tf.placeholder(dtype=tf.float32, shape=[m, n_out])

    def __generate_history_dict(self):
        hist = {'train': {'loss': [], 'accuracy': []},
                'test': {'loss': [], 'accuracy': []}}

        return hist

    def __train(self, train, test, batch_size, epochs, shuffle):
        hist = self.__generate_history_dict()

        X_train, Y_train = train
        X_test, Y_test = test
        m = Y_train.shape[0]

        if self.ckpt_file != None:
            saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for epoch in range(epochs):
                print('Epcoh {}/{}'.format(epoch+1, epochs))

                train_generator = batches_generator(X_train, Y_train, batch_size, shuffle)
                test_generator = batches_generator(X_test, Y_test, batch_size, shuffle)

                train_loss = []
                train_accuracy = []
                test_loss = []
                test_accuracy = []

                total_batches = m // batch_size
                if m % batch_size != 0:
                    total_batches += 1
                for X_train_, Y_train_ in tqdm(train_generator, total=total_batches):
                    _, l, a = sess.run([self.train_step, self.loss, self.accuracy], 
                                        feed_dict={self.layers[0]: X_train_,
                                                   self.Y: Y_train_})
                    train_loss.append(l)
                    train_accuracy.append(a)

                for X_test_, Y_test_ in test_generator:
                    l, a = sess.run([self.loss, self.accuracy], 
                                    feed_dict={self.layers[0]: X_test_,
                                               self.Y: Y_test_})
                    test_loss.append(l)
                    test_accuracy.append(a)

                hist['train']['loss'].append(np.mean(train_loss))
                hist['train']['accuracy'].append(np.mean(train_accuracy))
                hist['test']['loss'].append(np.mean(test_loss))
                hist['test']['accuracy'].append(np.mean(test_accuracy))

                if self.show_stats:
                    self.__print_stats(hist, epoch, epochs)


            if self.ckpt_file is not None:
                saver.save(sess, self.ckpt_file)

            return hist

    def __print_stats(self, hist, epoch, epochs):
        print('loss: {} - acc: {}'.format(hist['train']['loss'][-1], hist['train']['accuracy'][-1]))
        print('val_loss: {} - val_acc: {}'.format(hist['test']['loss'][-1], hist['test']['accuracy'][-1]))
        cols, _ = tuple(shutil.get_terminal_size((80, 20)))
        print('-'*cols)

    def __apply_loss(self):
        return LOSSES[self.loss_type](self.Y, self.layers[-1])

    def __get_suitable_accuracy(self):
        return METRICS[self.loss_type](self.Y, self.layers[-1])

    def add(self, layer):
        self.layers.append(layer)
        return layer

    def compile(self, optimizer='adam', loss='binary_entropy', 
                learning_rate=1e-2, show_stats=True, ckpt_file=None,
                device='cpu'):
        assert optimizer.lower() in list(OPTIMIZERS.keys())
        assert loss.lower() in list(LOSSES.keys())
        
        self.ckpt_file = ckpt_file
        self.loss_type = loss.lower()
        self.optimizer_type = optimizer.lower()
        self.__make_label_placeholder()

        self.accuracy = self.__get_suitable_accuracy()
        self.loss = self.__apply_loss()

        self.optimizer = OPTIMIZERS[self.optimizer_type](learning_rate=learning_rate)
        self.train_step = self.optimizer.minimize(self.loss)
        self.show_stats = show_stats
        self.device = device

        if self.device == 'cpu':
            os.environ['CUDA_VISIBLE_DEVICES'] = ''

    def fit(self, train, test=None, validation_split=0.1, epochs=1,
            shuffle=True, batch_size=32):
        assert isinstance(train, tuple)
        assert isinstance(test, tuple) or test is None

        # if test is None do Validation split
        self.do_val_split = test is None
        self.validation_split = validation_split

        X_train, Y_train = train
        if self.do_val_split:
            X_train, Y_train, X_test, Y_test = \
                        train_test_split(X_train, Y_train, shuffle, self.validation_split)
        else:
            X_test, Y_test = test

        return self.__train((X_train, Y_train), (X_test, Y_test), batch_size, epochs, shuffle)

    def predict(self, X, keep_prob=False, threshold=0.5):
        in_generator = batches_generator(X, batch_size=32)
        preds = []
        with tf.Session() as sess:
            for x in in_generator:
                preds.append(self.Y, feed_dict={self.layers[0]: x})
        preds = np.concatenate(preds)
        if keep_prob:
            return preds
        else:
            return self.__convert_to_one_hot(preds)
