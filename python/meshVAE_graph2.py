# coding: utf-8
import tensorflow as tf
import os
import numpy as np
import scipy.io as sio
from six.moves import xrange
import scipy.interpolate as interpolate
import h5py
import time
import random, pickle
from utils import *


class convMesh():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    def __init__(self, feature_file, FLAGS, logfolder):

        self.model = FLAGS.model
        self.lambda_generation = FLAGS.lambda_generation
        self.lambda_latent = FLAGS.lambda_latent
        self.lambda_r2 = FLAGS.lambda_r2
        self.sp_learning_rate = False
        self.lr = FLAGS.lr
        self.epoch_num = FLAGS.epoch_num
        self.layers = 3
        self.K = FLAGS.K
        self.hidden_dim = FLAGS.hidden_dim
        self.batch_size = FLAGS.batch_size
        self.use_pooling = FLAGS.use_pooling
        self.max_pooling = FLAGS.max_pooling
        self.useS = True
        self.result_min = -0.95
        self.result_max = 0.95
        self.vae_ablity = FLAGS.vae_ablity
        
        self.tb = False

        self.logfolder = logfolder

        self.feature, self.neighbour1, self.degree1, self.mapping1, self.neighbour2, self.degree2, \
         self.maxdemapping, self.meanpooling_degree, self.meandepooling_mapping, self.meandepooling_degree, \
         self.logrmin, self.logrmax, self.smin, self.smax, self.modelnum, self.pointnum1, self.pointnum2, \
         self.maxdegree1, self.maxdegree2, self.mapping1_col, self.L1, self.L2, self.cotw1, self.cotw2 \
         = load_data(feature_file, self.result_min, self.result_max, useS=self.useS, graphconv=True)

        if not self.useS:
            self.vertex_dim = 3
            self.finaldim = 3
        else:
            self.vertex_dim = 9
            self.finaldim = 9

        self.inputs = tf.placeholder(tf.float64, [None, self.pointnum1, self.vertex_dim], name='input_mesh')
        self.nb1 = tf.constant(self.neighbour1, dtype='int64', shape=[self.pointnum1, self.maxdegree1],
                               name='nb_relation1')
        self.nb2 = tf.constant(self.neighbour2, dtype='int64', shape=[self.pointnum2, self.maxdegree2],
                               name='nb_relation2')
        self.degrees1 = tf.constant(self.degree1, dtype='float64', shape=[self.pointnum1, 1], name='degrees1')
        self.degrees2 = tf.constant(self.degree2, dtype='float64', shape=[self.pointnum2, 1], name='degrees2')
        self.random = tf.placeholder(tf.float64, [None, self.hidden_dim], name='random_samples')
        self.cw1 = tf.constant(self.cotw1, dtype='float64', shape=[self.pointnum1, self.maxdegree1, 1], name='a/cw1')
        self.cw2 = tf.constant(self.cotw2, dtype='float64', shape=[self.pointnum2, self.maxdegree2, 1], name='a/cw2')
        self.Laplace1 = self.L1
        if self.use_pooling:
            self.Laplace2 = self.L2
        else:
            self.Laplace2 = self.L1

        if self.use_pooling and self.max_pooling:
            self.mappingpooling1 = tf.constant(self.mapping1, dtype='int64', shape=[self.pointnum2, self.mapping1_col],
                                               name='mapping')
            self.mappingdepooling1 = tf.constant(self.maxdemapping, dtype='int64', shape=[self.pointnum1, 1],
                                                 name='demapping')
        elif self.use_pooling and not self.max_pooling:
            self.mappingpooling1 = tf.constant(self.mapping1, dtype='int64', shape=[self.pointnum2, self.mapping1_col],
                                               name='mapping')
            self.mappingdepooling1 = tf.constant(self.meandepooling_mapping, dtype='int64', shape=[self.pointnum1, 1],
                                                 name='demapping')
            self.mean_pl_degree = tf.constant(self.meanpooling_degree, dtype='float64', shape=[self.pointnum2, 1],
                                              name='mapping_degree')
            self.mean_depl_degree = tf.constant(self.meandepooling_degree, dtype='float64', shape=[self.pointnum1, 1],
                                                name='demapping_degree')
        else:
            self.nb2, self.degrees2 = self.nb1, self.degrees1
            self.pointnum2, self.cw2 = self.pointnum1, self.cw1
            print('we don\'t use pooling!!')

        self.enc_w = []
        self.dec_w = []
        for i in range(self.layers):
            if i == self.layers - 1:
                enc_weight = tf.get_variable("encoder/conv_weight"+str(i+1), [self.vertex_dim * self.K, self.finaldim], tf.float64,
                                  tf.random_normal_initializer(stddev=0.02))
                dec_weight = tf.get_variable("decoder/conv_weight"+str(i+1), [self.vertex_dim * self.K, self.finaldim], tf.float64,
                                  tf.random_normal_initializer(stddev=0.02))
            else:
                enc_weight = tf.get_variable("encoder/conv_weight"+str(i+1), [self.vertex_dim * self.K, self.vertex_dim], tf.float64,
                                  tf.random_normal_initializer(stddev=0.02))
                dec_weight = tf.get_variable("decoder/conv_weight"+str(i+1), [self.vertex_dim * self.K, self.vertex_dim], tf.float64,
                                  tf.random_normal_initializer(stddev=0.02))
            self.enc_w.append(enc_weight)
            self.dec_w.append(dec_weight)


        self.meanpara = tf.get_variable("encoder/mean_weights", [self.pointnum2 * self.finaldim, self.hidden_dim],
                                        tf.float64, tf.random_normal_initializer(stddev=0.02))
        self.stdpara = tf.get_variable("encoder/std_weights", [self.pointnum2 * self.finaldim, self.hidden_dim],
                                       tf.float64, tf.random_normal_initializer(stddev=0.02))

        # train
        self.z_mean, self.z_stddev = self.encoder(self.inputs, train=True)
        self.guessed_z = self.z_mean + self.z_stddev * tf.random_normal(tf.shape(self.z_mean), 0, 1, dtype=tf.float64)
        self.generated_mesh_train = self.decoder(self.guessed_z, train=True)

        # test
        self.z_mean_test, self.z_stddev_test = self.encoder(self.inputs, train=False)
        self.guessed_z_rebuild = self.z_mean_test
        self.generated_mesh_rebuild = self.decoder(self.guessed_z_rebuild, train=False)

        # generation
        self.test_mesh = self.decoder(self.random, train=False)

        self.generation_loss = self.lambda_generation * tf.reduce_mean(tf.reduce_sum(tf.pow(self.inputs - self.generated_mesh_train, 2.0), [1, 2]))

        self.latent_loss = self.lambda_latent * tf.reduce_mean(0.5 * tf.reduce_sum(
            tf.square(self.z_mean) + tf.square(self.z_stddev) - tf.log(1e-8 + tf.square(self.z_stddev)) - 1, [1]))

        self.valid_loss = self.lambda_generation * tf.reduce_mean(tf.reduce_sum(tf.pow(self.inputs - self.generated_mesh_rebuild, 2.0), [1, 2]))

        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope='encoder') + tf.get_collection(
            tf.GraphKeys.REGULARIZATION_LOSSES, scope='decoder')

        self.r2_loss = sum(reg_losses) + tf.nn.l2_loss(self.enc_w) + tf.nn.l2_loss(self.dec_w) + tf.nn.l2_loss(self.meanpara) + tf.nn.l2_loss(self.stdpara)
        self.r2_loss = self.r2_loss * self.lambda_r2

        self.loss = self.generation_loss + self.latent_loss + self.r2_loss

        self.global_step = tf.Variable(0, trainable=False)
        if self.sp_learning_rate:
            new_learning_rate = tf.train.exponential_decay(self.lr, self.global_step, 3000, 0.5, staircase=True)
            self.optimizer = tf.train.AdamOptimizer(new_learning_rate).minimize(self.loss)
        else:
            self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        self.saver_best = tf.train.Saver(max_to_keep = None)
        self.saver_vae = tf.train.Saver(max_to_keep = 3)
        tf.summary.scalar('loss_all', self.loss)
        tf.summary.scalar('loss_generation', self.generation_loss)
        tf.summary.scalar('loss_latent', self.latent_loss)
        tf.summary.scalar('loss_r2', self.r2_loss)
        self.merge_summary = tf.summary.merge_all()

        self.checkpoint_dir = self.logfolder
        if os.path.exists(self.logfolder + '/log.txt'):
            self.log_file = open(self.logfolder + '/log.txt', 'a')
        else:
            self.log_file = open(self.logfolder + '/log.txt', 'w')

        if os.path.exists(logfolder + '/result.txt'):
            self.sim_log_file = open(logfolder + '/result.txt', 'a')
        else:
            self.sim_log_file = open(logfolder + '/result.txt', 'w')

    # functions
    def get_conv_weights(self, input_dim, output_dim, name='convweight'):
        with tf.variable_scope(name) as scope:
            n = tf.get_variable("nb_weights", [input_dim, output_dim], tf.float64,
                                tf.random_normal_initializer(stddev=0.02))
            v = tf.get_variable("vertex_weights", [input_dim, output_dim], tf.float64,
                                tf.random_normal_initializer(stddev=0.02))

            return n, v

    def encoder(self, input_feature, train=True, reuse=False):
        with tf.variable_scope("encoder") as scope:
            scope.set_regularizer(tf.contrib.layers.l2_regularizer(scale=1.0))
            if not train or reuse:
                train = False
                reuse = True
                scope.reuse_variables()

            conv1 = graph_conv2(input_feature, self.Laplace1, self.vertex_dim, self.enc_w[0], self.K, name='conv1', training=train)
            conv1 = graph_conv2(conv1, self.Laplace1, self.vertex_dim, self.enc_w[1], self.K, name='conv2', training=train)

            if self.use_pooling and self.max_pooling:
                conv1 = mesh_max_pooling(conv1, self.mappingpooling1)
            elif self.use_pooling and not self.max_pooling:
                conv1 = mesh_mean_pooling(conv1, self.mappingpooling1, self.mean_pl_degree)
            else:
                conv1 = conv1

            conv1 = graph_conv2(conv1, self.Laplace2, self.finaldim, self.enc_w[2], self.K, name='conv3', training=train,
                                bn=False)
            x0 = tf.reshape(conv1, [tf.shape(conv1)[0], self.pointnum2 * self.finaldim])
            mean = linear1(x0, self.meanpara, self.hidden_dim, 'mean')
            stddev = linear1(x0, self.stdpara, self.hidden_dim, 'stddev')
            stddev = tf.sqrt(2 * tf.nn.sigmoid(stddev))

            return mean, stddev

    def decoder(self, latent_tensor, train=True, reuse=False):
        with tf.variable_scope("decoder") as scope:
            scope.set_regularizer(tf.contrib.layers.l2_regularizer(scale=1.0))
            if not train or reuse:
                train = False
                reuse = True
                scope.reuse_variables()

            l1 = linear1(latent_tensor, tf.transpose(self.meanpara), self.pointnum2 * self.finaldim, 'mean')
            l2 = tf.reshape(l1, [tf.shape(l1)[0], self.pointnum2, self.finaldim])
            conv1 = graph_conv2(l2, self.Laplace2, self.vertex_dim, self.dec_w[2], self.K, name='conv4', training=train)
            if self.use_pooling and self.max_pooling:
                conv1 = mesh_max_depooling(conv1, self.mappingdepooling1)
            elif self.use_pooling and not self.max_pooling:
                conv1 = mesh_mean_depooling(conv1, self.mappingdepooling1, self.mean_depl_degree)
            else:
                conv1 = conv1

            conv1 = graph_conv2(conv1, self.Laplace1, self.vertex_dim, self.dec_w[1], self.K, name='conv5', training=train)

            conv1 = graph_conv2(conv1, self.Laplace1, self.vertex_dim, self.dec_w[0], self.K, name='conv6',
                                training=train, bn=False)

        return conv1

    def train(self):
        with tf.Session(config=self.config) as self.sess:
            if not os.path.exists(self.checkpoint_dir):
                os.makedirs(self.checkpoint_dir)

            tf.global_variables_initializer().run()

            could_load_vae, checkpoint_counter_vae = self.load(self.checkpoint_dir)

            if (could_load_vae and checkpoint_counter_vae < self.epoch_num):
                self.start_step_vae = checkpoint_counter_vae
            else:
                self.start_step_vae = 0

            self.write = tf.summary.FileWriter(self.logfolder + '/logs/', self.sess.graph)

            rng = np.random.RandomState(23456)

            file_name = 'id'+str(self.vae_ablity)+'.dat'
            if os.path.isfile(file_name):
                id = pickle.load(open(file_name, 'rb'))
                id.show()
                Ia = id.Ia
                Ib = id.Ib
            else:
                Ia = np.arange(len(self.feature))
                Ia = random.sample(list(Ia), int(len(self.feature) * (1 - self.vae_ablity)))
                Ib = Ia
                id = Id(Ia, Ib)
                id.show()
                f = open(file_name, 'wb')
                pickle.dump(id, f, pickle.HIGHEST_PROTOCOL)
                f.close()
                id = pickle.load(open(file_name, 'rb'))
                id.show()

            self.C_Ia = list(set(np.arange(len(self.feature))).difference(set(Ia)))
            valid_best = float('inf')

            batch_size = self.batch_size
            for epoch in xrange(self.start_step_vae, self.epoch_num):
                timecurrent = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
                rng.shuffle(Ia)
                train_feature = self.feature[Ia]
                train_num = len(train_feature)

                for bidx in xrange(0, train_num//batch_size + 1):
                    train_feature_batch = train_feature[bidx*batch_size:min(train_num, bidx*batch_size+batch_size)]
                    if len(train_feature_batch) == 0:
                        continue
                    random_batch = np.random.normal(loc=0.0, scale=1.0, size=(len(train_feature_batch), self.hidden_dim))

                    _, cost_generation, cost_latent, cost_r2 = self.sess.run([self.optimizer, self.generation_loss, self.latent_loss, self.r2_loss], feed_dict={self.inputs: train_feature_batch, self.random: random_batch})

                    print("%s Epoch|Batch: [%4d|%4d]generation_loss: %.8f, latent_loss: %.8f r2_loss: %.8f" % (
                        timecurrent, epoch + 1, bidx+1, cost_generation, cost_latent, cost_r2))
                    self.log_file.write("%s Epoch|Batch: [%4d|%4d]generation_loss: %.8f, latent_loss: %.8f, r2_loss: %.8f\n" % (
                        timecurrent, epoch + 1, bidx+1, cost_generation, cost_latent, cost_r2))

                test_num = len(self.C_Ia)
                valid_generation = 0
                for bidx in xrange(0, test_num//batch_size + 1):
                    dxb = self.C_Ia[bidx*batch_size:min(test_num, bidx*batch_size+batch_size)]
                    test_feature_batch = self.feature[dxb]
                    valid_generation_batch = self.valid_loss.eval({self.inputs: test_feature_batch})
                    valid_generation += valid_generation_batch
                print("%s Epoch: [%4d]valid_loss: %.8f" % (timecurrent, epoch + 1, valid_generation))

                if np.mod(epoch + 1, 100) == 0:

                    if valid_generation < valid_best:
                        valid_best = valid_generation
                        print('Save best!')
                        self.log_file.write("Save Best! Epoch: %4d\n" % (epoch + 1))
                        self.sim_log_file.write("%d %.8f %.8f %.8f %.8f\n" % (1, cost_generation, cost_latent, valid_generation, cost_r2))
                        self.saver_best.save(self.sess, self.logfolder + '/' + 'convMesh_validbest.model')
                    else:
                        self.sim_log_file.write("%d %.8f %.8f %.8f %.8f\n" % (0, cost_generation, cost_latent, valid_generation, cost_r2))

                    self.saver_vae.save(self.sess, self.logfolder + '/' + 'convMesh.model', global_step=epoch + 1)

                    rebuild_mesh = []
                    for bidx in xrange(0, len(self.feature)//batch_size + 1):
                        rebuild_batch = self.generated_mesh_rebuild.eval({self.inputs: self.feature[bidx*batch_size:min(len(self.feature), bidx*batch_size+batch_size)]})
                        rebuild_mesh_batch = recover_data(rebuild_batch, self.logrmin, self.logrmax, self.smin, self.smax, self.result_min, self.result_max, self.useS)
                        if bidx == 0:
                            rebuild_mesh = rebuild_mesh_batch
                        else:
                            rebuild_mesh = np.concatenate((rebuild_mesh, rebuild_mesh_batch), axis=0)

                    savefile = h5py.File(self.logfolder + '/' + 'rebuild' + str(epoch + 1) + '.h5', 'w')
                    savefile['test_mesh'] = rebuild_mesh
                    savefile['valid_id'] = self.C_Ia
                    savefile.close()

                if self.tb and (epoch + 1) % 20 == 0:
                    s = self.sess.run(self.merge_summary, feed_dict={self.inputs: self.feature, self.random: random_batch})
                    self.write.add_summary(s, epoch)

        self.log_file.close()
        self.sim_log_file.close()
        return

    def interpola(self, restore, begin_id, end_id, interval, foldername):
        with tf.Session(config=self.config) as self.sess:
            self.saver_vae.restore(self.sess, restore)
            x = np.zeros([2, self.pointnum1, self.vertex_dim])
            x[0, :, :] = self.feature[begin_id, :, :]
            x[1, :, :] = self.feature[end_id, :, :]

            random_np = self.guessed_z_rebuild.eval({self.inputs: x})
            random2_intpl = interpolate.griddata(
                np.linspace(0, 1, len(random_np) * 1), random_np,
                np.linspace(0, 1, interval), method='linear')

            if not os.path.isdir(foldername):
                os.mkdir(foldername)

            test = self.sess.run([self.test_mesh], feed_dict={self.random: random2_intpl})[0]

            test = recover_data(test, self.logrmin, self.logrmax, self.smin, self.smax, self.result_min, self.result_max, self.useS)

            name = foldername + '/intlp_test' + str(begin_id) + '_' + str(end_id) + '.h5'
            print(name)
            f = h5py.File(name, 'w')
            f['test_mesh'] = test
            f['latent_z'] = random_np
            f.close()

        return

    def embedding(self, restore):
        with tf.Session(config=self.config) as self.sess:
            self.saver_vae.restore(self.sess, restore)

            meanemb, stddev = self.sess.run([self.z_mean_test, self.z_stddev_test], feed_dict={self.inputs: self.feature})
            if not os.path.isdir(self.logfolder + '/embdata'):
                os.mkdir(self.logfolder + '/embdata')
            name = self.logfolder + '/embdata/embedding_' + model + '.h5'
            f = h5py.File(name, 'w')
            f['meanemb'] = meanemb
            f['stddev'] = stddev
            f.close()

        return

    def model_dir(self, model_name, dataset_name):
        return "{}_{}_{}".format(model_name, dataset_name, hidden_dim)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        # checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)
        saver = self.saver_vae

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)

            saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0  # model = convMESH()

    def random_generate(self, restore, gennum, foldername):
        with tf.Session(config=self.config) as self.sess:
            self.saver_vae.restore(self.sess, restore)

            random_batch = np.random.normal(loc=0.0, scale=1.0, size=(gennum, self.hidden_dim))
            if not os.path.isdir(foldername):
                os.mkdir(foldername)
            test = self.sess.run([self.test_mesh], feed_dict={self.random: random_batch})[0]
            test = recover_data(test, self.logrmin, self.logrmax, self.smin, self.smax, self.result_min, self.result_max, self.useS)

            file_name = 'id' + str(vae_ablity) + '.dat'
            id = pickle.load(open(file_name, 'rb'))
            Ia = id.Ia

            name = foldername + '/random_gen' + str(gennum) + '.h5'
            print(name)
            f = h5py.File(name, 'w')
            f['test_mesh'] = test
            f['train_id'] = Ia
            f['latent_z'] = random_batch
            f.close()

        return
