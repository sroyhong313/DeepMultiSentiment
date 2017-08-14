import _pickle
from model import Model
import process_data_sst1
import tensorflow as tf

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

FLAGS =tf.flags.FLAGS
FLAGS._parse_flags()
print ("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print ("")

# Data Preparation
# ==========================================================
print ("Loading Data...")
process_data_sst1.process_data("data/processed/sst1.p")
x = _pickle.load(open("data/processed/sst1.p", "rb"))

# Training
# ==========================================================
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Code that operates on the default graph and session comes here
        revs, embedding, W2, word_idx_map, vocab, max_length = x[0], x[1], x[2], x[3], x[4], x[5]
        if FLAGS.random:
            embedding = W2
        cnn = Model(
            embedding_size=FLAGS.embedding_dim,
            vocab_size=len(vocab) + 1,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            seq_length=max_length + 2 * 4,
            num_classes=FLAGS.num_classes,
            l2_reg_lambda=FLAGS.l2_reg_lambda,
            trainable=not FLAGS.static
            )
        train_data, train_labels, test_data, test_labels = self.create_data(revs, word_idx_map, max_length, self.num_classes)

        # optimize loss function
        # By defining a global_step variable and passing it to the optimizer
        # we allow TensorFlow handle the counting of training steps for us.
        # The global step will be automatically incremented by one every time you execute train_op.
        global_step = tf.Variable(0, name="global_step", trainable=False)
        learning_rate_decay = tf.placeholder(tf.float32)
        learning_rate = self.learning_rate_decay * 1e-3
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
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
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph_def)

        # Test summaries (evaluation)
        test_summary_op = tf.summary.merge([loss_summary, acc_summary])
        test_summary_dir = os.path.join(out_dir, "summaries", "test")
        test_summary_writer = tf.summary.FileWriter(test_summary_dir, sess.graph_def)

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
        sess.run(self.embedding_init, feed_dict={self.embedding_placeholder: embedding})


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
        def test_step(x_batch, y_batch, writer=None):
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

        def get_idx_from_sent(self, sent, word_idx_map, max_length, filter_h=5):
            """
            Transforms sentence into a list of indices. Pad with zeroes.
            """
            x = []
            pad = filter_h - 1
            for i in xrange(pad):
                x.append(0)
            words = sent.split()
            for word in words:
                if word in word_idx_map:
                    x.append(word_idx_map[word])
            while len(x) < max_length + 2 * pad:
                x.append(0)
            return x

        def make_idx_data_cv(self, revs, word_idx_map, max_length, filter_h=5):
            """
            Transforms sentences into a 2-d matrix.
            """
            train_data, test_data = [], []
            train_labels, test_labels = [], []
            for rev in revs:
                sent = self.get_idx_from_sent(rev["text"], word_idx_map, max_length, filter_h)
                label = [0] * self.FLAGS.num_classes
                label[rev["y"]] = 1
                if rev["split"] == 1:
                    test_data.append(sent)
                    test_labels.append(label)
                else:
                    train_data.append(sent)
                    train_labels.append(label)
            train_data = np.array(train_data, dtype="int")
            test_data = np.array(test_data, dtype="int")
            train_labels = np.asarray(train_labels, dtype=np.float32)
            test_labels = np.asarray(test_labels, dtype=np.float32)

            return [train_data, train_labels, test_data, test_labels]

        def create_data(self, revs, word_idx_map, max_length, num_classes):
            train_data, train_labels, test_data, test_labels = self.make_idx_data_cv(revs, word_idx_map, max_length, 5)

            shuffle_indices = np.random.permutation(np.arange(len(train_data)))
            train_data = train_data[shuffle_indices]
            train_labels = train_labels[shuffle_indices]

            return train_data, train_labels, test_data, test_labels


        # Training steps
        batch_size = FLAGS.batch_size
        num_epochs = FLAGS.num_epochs
        steps_per_epoch = int(len(train_labels) / batch_size)
        num_steps = steps_per_epoch * num_epochs
        epoch_num = 0
        learning_rate_decay = 1
        for step in range(num_steps):
            # Shuffle the data in each epoch
            if (step % steps_per_epoch == 0):
                shuffle_indices = np.random.permutation(np.arange(len(train_data)))
                train_data = train_data[shuffle_indices]
                train_labels = train_labels[shuffle_indices]
                print("epoch number %d" % epoch_num)
                # Get test results on each epoch
                test_step(train_data, train_labels, writer=test_summary_writer)
                current_step = tf.train.global_step(sess, global_step)
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                # feed_dict = {self.input_x: test_data, self.input_y: test_labels, self.dropout_keep_prob: 1.0}
                # accuracy_out = sess.run(self.accuracy, feed_dict=feed_dict)
                # print('Test accuracy: %.3f' % accuracy_out)
                # On each 8 epochs we decay the learning rate
                if(epoch_num == 8):
                        learning_rate_decay = learning_rate_decay*0.5
                if(epoch_num == 16):
                        learning_rate_decay = learning_rate_decay*0.1
                epoch_num += 1
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
            batch_data = train_data[offset:(offset + batch_size), :]
            batch_labels = train_labels[offset:(offset + batch_size), :]
            # Setting all placeholders
            train_step(batch_data, batch_labels)
            # feed_dict = {self.input_x: batch_data, self.input_y: batch_labels, self.dropout_keep_prob: 0.5,
            #              self.learning_rate_decay: learning_rate_decay}
            # _, l, accuracy_out = sess.run(
            #     [self.optimizer, self.loss, self.accuracy], feed_dict=feed_dict)
        # Testing
        test_step(x_batch, y_batch)
        feed_dict = {self.input_x: test_data, self.input_y: test_labels, self.dropout_keep_prob: 1.0}
        accuracy_out = sess.run(self.accuracy, feed_dict=feed_dict)
        print('Test accuracy: %.3f' % accuracy_out)
