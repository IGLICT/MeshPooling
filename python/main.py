import os
import tensorflow as tf
import meshVAE_graph2 as model

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("gpu", '0', "gpu id")
tf.flags.DEFINE_string("model", 'dense_scape', "model name")
tf.flags.DEFINE_float("lambda_generation", 40, "lambda of generation error")
tf.flags.DEFINE_float("lambda_latent", 3, "lambda of KL divergence")
tf.flags.DEFINE_float("lambda_r2", 1, "lambda of l2 regularization")
tf.flags.DEFINE_float("lr", 0.001, "learning rate")
tf.flags.DEFINE_integer("epoch_num", 8000, "epoch number")
tf.flags.DEFINE_integer("hidden_dim", 128, "latent dimension")
tf.flags.DEFINE_boolean("use_pooling", True, "use pooling or not")
tf.flags.DEFINE_boolean("max_pooling", False, "max pooling or mean pooling")
tf.flags.DEFINE_float("vae_ablity", 0.5, "ratio of models used for validation")
tf.flags.DEFINE_integer("K", 3, "graph convolution parameter")
tf.flags.DEFINE_integer("batch_size", 50, "batch size")

feature_file = 'vertFeaturepooling' + FLAGS.model + '.mat'

if FLAGS.use_pooling and FLAGS.max_pooling:
    logfolder = './' + FLAGS.model + '_' + str(FLAGS.lambda_generation) + '_' + \
                      str(FLAGS.lambda_latent) + '_maxpooling_' + str(FLAGS.vae_ablity) + '_K=' + str(FLAGS.K)
elif FLAGS.use_pooling and not FLAGS.max_pooling:
    logfolder = './' + FLAGS.model + '_' + str(FLAGS.lambda_generation) + '_' + \
                      str(FLAGS.lambda_latent) + '_meanpooling_' + str(FLAGS.vae_ablity) + '_K=' + str(FLAGS.K)
else:
    logfolder = './' + FLAGS.model + '_' + str(FLAGS.lambda_generation) + '_' + \
                      str(FLAGS.lambda_latent) + '_nopooling_' + str(FLAGS.vae_ablity) + '_K=' + str(FLAGS.K)

if not os.path.isdir(logfolder):
    os.mkdir(logfolder)

os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu
convmesh = model.convMesh(feature_file, FLAGS, logfolder)
convmesh.train()

# interpolation
# inter_begin_id = 7
# inter_end_id = 14
# interval = 30
# convmesh.interpola(logfolder + '/convMesh.model-' + str(FLAGS.epoch_num), inter_begin_id, inter_end_id, interval, logfolder + '/itlp')

# random generation
# gen_num = 100
# convmesh.random_generate(logfolder + '/convMesh.model-' + str(FLAGS.epoch_num), gen_num, logfolder + '/random')