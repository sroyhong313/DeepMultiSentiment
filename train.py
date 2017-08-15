import _pickle
import tensorflow as tf

from model import TextCNN
from utils import process_data, create_data, batch_iter
import time
import datetime
import os
# Parameters
# ==========================================================

tf.flags.DEFINE_boolean("random",False,"Initialize with random word embeddings (default: False)")
tf.flags.DEFINE_boolean("static",False,"Keep the word embeddings static (default: False)")

tf.flags.DEFINE_integer("num_classes", 5, "Number of output classes (default: 5 (SST-1))")
tf.flags.DEFINE_integer("k-fold", 1, "Increase k-fold to 10 if no dev/test set is availabe (default:1)")
tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of Character Embedding (default: 300 [Google W2Vec Dim])")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 100, "Number of filters per filter size (default:100)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.15, "Lambda value of L2-reg (default: 0.15)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")

tf.flags.DEFINE_integer("num_epochs", 25, "Number of training epochs (Default: 25)")
tf.flags.DEFINE_integer("batch_size", 50, "Batch Size (default: 50)")
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_integer("evaluate_every", 300, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_integer("checkpoint_every", 300, "Save model after this many steps (default: 100)")

FLAGS =tf.flags.FLAGS
FLAGS._parse_flags()
print ("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print ("")

# Data Preparation
# ==========================================================
print ("Loading Data...")
process_data("data/processed/sst1.p")
x = _pickle.load(open("data/processed/sst1.p", "rb"))
revs, embedding, W2, word_idx_map, vocab, max_length = x[0], x[1], x[2], x[3], x[4], x[5]
x_train, y_train, x_dev, y_dev = create_data(revs, word_idx_map, max_length, FLAGS.num_classes)


# Training
# ==========================================================
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Code that operates on the default graph and session comes here
        if FLAGS.random:
            embedding = W2
        cnn = TextCNN(
            seq_length=x_train.shape[1],
            num_classes=y_train.shape[1],
            vocab_size=len(vocab) + 1,
            embedding_size=FLAGS.embedding_dim,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            l2_reg_lambda=FLAGS.l2_reg_lambda)

        # optimize loss function
        # By defining a global_step variable and passing it to the optimizer
        # we allow TensorFlow handle the counting of training steps for us.
        # The global step will be automatically incremented by one every time you execute train_op.
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print ("Writing to {}\n".format(out_dir))

        # Summaries for loss & accuracy
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

        # Train summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Test summaries (evaluation)
        test_summary_op = tf.summary.merge([loss_summary, acc_summary])
        test_summary_dir = os.path.join(out_dir, "summaries", "test")
        test_summary_writer = tf.summary.FileWriter(test_summary_dir, sess.graph)

        # Checkpointing
        # -- saving the parameters of your model to restore them later on.
        # -- Checkpoints can be used to continue training at a later point,
        # -- or to pick the best parameters setting using early stopping
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        # TF assumes this directory already exists so we need to create it
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        sess.run(tf.global_variables_initializer())
        sess.run(cnn.embedding_init, feed_dict={cnn.embedding_placeholder: embedding})

        # methods for batch separation & training here
        # ==================================================
        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print ("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

        # Evaluate the loss & accuracy on an arbitrary data set without dropout nor training ops
        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a test set
            """
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.dropout_keep_prob: 1.0
            }
            step, summaries, loss, accuracy = sess.run(
                [global_step, test_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print ("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)

        batches = batch_iter(
            list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)

        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print ("\nEvaluation:")
                dev_step(x_dev, y_dev, writer=test_summary_writer)
                print ("")
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print ("Saved model checkpoint to {}\n".format(path))