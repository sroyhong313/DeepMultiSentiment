# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np


class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling, and softmax layer
    """
    def __init__(self, embedding_size, vocab_size, filter_sizes, num_filters, seq_length,
                    num_classes, l2_reg_lambda=0.15):
        """
        seq_length:     length of sentences. Padded st all have same length (59 for this dataset)
        num_classes:    # classes in output layer, two in this sentiment analysis case (pos & neg)
        vocab_size:     size of vocabulary. Needed to define embedding layer which will have shape
                        [vocab_size, embedding_size]
        embedding_size: Dimensionality of the embedding
        filter_sizes:   # words we want our convolutional filters to cover.
                        Will have num_filters for each size specified here.
                        [3, 4, 5] ==> filters that slide over 3, 4, 5 words respectively, for a total of
                        3 * num_filters filters
        num_filters:    # filters per filter size (see filter_sizes)
        """

        self.input_x = tf.placeholder(tf.int32, [None, seq_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization
        l2_loss = tf.constant(0.0)

        # The first layer we define is the embedding layer, which maps vocabulary word indices into
        # low-dimensional vector representations.
        # It’s essentially a lookup table that we learn from data

        # TensorFlow’s convolutional conv2d operation expects a 4-dimensional tensor with dimensions corresponding to batch,
        # width, height and channel. The result of our embedding doesn’t contain the channel dimension, so we add it manually,
        # through expand_dims, leaving us with a layer of shape [None, sequence_length, embedding_size, 1]
        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            W = tf.Variable(tf.random_uniform([vocab_size, embedding_size],
                                     -1.0, 1.0), name="W")
            self.embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_size])
            self.embedding_init = W.assign(self.embedding_placeholder)
            self.embedded_sent = tf.nn.embedding_lookup(W, self.input_x)
            self.embedded_sent_expanded = tf.expand_dims(self.embedded_sent, -1)

        # Create a convolution + max_pooling layer for each filter size
        # Because each convolution produces tensors of different shapes we need to
        # iterate through them, create a layer for each of them, and then merge the
        # results into one big feature vector.
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                # Filter_shape: Must have the same type as input. A 4-D tensor of shape [filter_height, filter_width, in_channels, out_channels]
                # With the default format "NHWC", the data is stored (output) in the order of: [batch, height, width, channels]
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                # Weights for each filter
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                # Bias for each filter
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                # The actual convolution
                conv = tf.nn.conv2d(
                    self.embedded_sent_expanded,
                    W,
                    strides=[1,1,1,1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Max Pooling over the outputs
                # VALID padding means that we slide the filter over our sentence without padding the edges,
                # performing a narrow convolution that gives us an output of
                # shape [1, sequence_length - filter_size + 1, 1, 1].
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, seq_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            # l2 loss
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")