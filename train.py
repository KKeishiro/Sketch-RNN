import os
import os.path as osp
import six
import json
import random
import numpy as np
import tensorflow as tf
import keras
from keras.models import Model
from keras.layers import (Input, Reshape, RepeatVector, concatenate)
from keras.callbacks import (ModelCheckpoint, EarlyStopping, CSVLogger,
                            LearningRateScheduler, TensorBoard, TerminateOnNaN)
from keras.utils import plot_model
from keras import backend as K
import sketch_rnn_model
import utils


tf.logging.set_verbosity(tf.logging.INFO)

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'data_dir',
    './data',
    'The directory in which to find the dataset specified in model hparams. '
    'If data_dir starts with "http://" or "https://", the file will be fetched '
    'remotely.')
tf.app.flags.DEFINE_string(
    'log_root', './models/default',
    'Directory to store model checkpoints, tensorboard.')
tf.app.flags.DEFINE_boolean(
    'resume_training', False,
    'Set to true to load previous checkpoint')
tf.app.flags.DEFINE_string(
    'hparams', '',
    'Pass in comma-separated key=value pairs such as '
    '\'kl_weight=1.0,decay_rate=0.99\' '
    '(no whitespace) to be read into the HParams object defined in model.py')
tf.app.flags.DEFINE_string(
    'weights', '',
    'File to load weights.')


def train():
  model_params = sketch_rnn_model.get_default_hparams()
  if FLAGS.hparams:
    model_params.parse(FLAGS.hparams)
  tf.logging.info('Hyperparams:')
  for key, val in six.iteritems(model_params.values()):
    tf.logging.info('%s = %s', key, str(val))
  tf.logging.info('Loading data files.')
  datasets = utils.load_dataset(FLAGS.data_dir, model_params)

  train_set = datasets[0]
  valid_set = datasets[1]
  test_set = datasets[2]
  model_params = datasets[3]

  model = sketch_rnn_model.SketchRNN(model_params)

  # Write config file to json file.
  tf.gfile.MakeDirs(FLAGS.log_root)
  with tf.gfile.Open(
      os.path.join(FLAGS.log_root, 'model_config.json'), 'w') as f:
    json.dump(model_params.values(), f, indent=True)

  hps = model.hps

  # Returns stroke-5 format
  x = train_set.pad_strokes()
  # vectors of strokes to be fed to encoder
  # this is also used as the target/expected vectors of strokes
  encoder_inputs = x[:, 1:hps.max_seq_len + 1, :]
  # vectors of strokes to be fed to decoder (same as above, but lagged behind
  # one step to include initial dummy value of (0, 0, 1, 0, 0))
  decoder_inputs = x[:, :hps.max_seq_len, :]

  x_val = valid_set.pad_strokes()
  encoder_inputs_val = x_val[:, 1:hps.max_seq_len + 1, :]
  decoder_inputs_val = x_val[:, :hps.max_seq_len, :]

  # encoder, sketch_rnn = model.build_model
  model.model_compile(model.sketch_rnn_model)
  plot_model(model.sketch_rnn_model, to_file='model.png', show_shapes=True)

  # define callbacks settings
  checkpointer = ModelCheckpoint(
                  filepath=osp.join(FLAGS.log_root, 'model.ep{epoch:02d}.hdf5'),
                  monitor='val_loss',
                  save_best_only=True,
                  verbose=1)
  tensorboard = TensorBoard(log_dir='./logs')
  csvlogger = CSVLogger('training.log')
  early_stopping = EarlyStopping(monitor='val_loss',
                                  verbose=1,
                                  patience=3)
  lr_scheduler = LearningRateScheduler(
      lambda epoch: ((hps.learning_rate - hps.min_learning_rate) *
                            (hps.decay_rate)**epoch + hps.min_learning_rate)
                            )

  if FLAGS.resume_training:
    assert FLAGS.weights is not '', 'please set weights file.'
    model.sketch_rnn_model.load_weights(FLAGS.weights)

  model.sketch_rnn_model.fit([encoder_inputs, decoder_inputs],
                      encoder_inputs,
                      batch_size=hps.batch_size,
                      epochs=hps.num_epochs,
                      verbose=1,
                      callbacks=[checkpointer, tensorboard, csvlogger,
                                early_stopping, lr_scheduler, TerminateOnNaN()],
                      validation_data=(
                        [encoder_inputs_val, decoder_inputs_val],
                        encoder_inputs_val)
                      )

# sampling model ---------------------------------------------------------------
def sample(model, hps, weights, seq_len=250, temperature=1.0,
            greedy_mode=False, z=None):
  """Samples a sequence from a pre-trained model."""

  def adjust_temp(pi_pdf, temp):
    pi_pdf = np.log(pi_pdf) / temp
    pi_pdf -= pi_pdf.max()
    pi_pdf = np.exp(pi_pdf)
    pi_pdf /=pi_pdf.sum()
    return pi_pdf

  def get_pi_idx(x, pdf, temp=1.0, greedy=False):
    """Samples from a pdf, optionally greedily."""
    if greedy:
      return np.argmax(pdf)
    pdf = adjust_temp(np.copy(pdf), temp)
    accumulate = 0
    for i in range(0, pdf.size):
      accumulate += pdf[i]
      if accumulate >= x:
        return i
    tf.logging.info('Error with smpling ensemble.')
    return -1

  def sample_gaussian_2d(mu1, mu2, s1, s2, rho, temp=1.0, greedy=False):
    if greedy:
      return mu1, mu2
    mean = [mu1, mu2]
    s1 *= temp * temp
    s2 *= temp * temp
    cov = [[s1 * s1, rho * s1 * s2], [rho * s1 * s2, s2 * s2]]
    x = np.random.multivariate_normal(mean, cov, 1)
    return x[0][0], x[0][1]

  # load model
  model.sketch_rnn_model.load_weights(weights)

  prev_x = np.zeros((1, 1, 5), dtype=np.float32)
  prev_x[0, 0, 2] = 1 # initially, we want to see beginning of new stroke
  if z is None:
    z = np.random.randn(1, hps.z_size) # not used if unconditional

  batch_z = Input(shape=(hps.z_size,)) # (1, z_size)
  initial_state = model.initial_state_layer(batch_z)
  # (1, dec_rnn_size * 2)

  decoder_input = Input(shape=(1, 5)) # (1, 1, 5)
  overlay_x = RepeatVector(1)(batch_z) # (1,1, z_size)
  actual_input_x = concatenate([decoder_input, overlay_x], 2)
  # (1, 1, 5 + z_size)

  decoder_h_input = Input(shape=(hps.dec_rnn_size, ))
  decoder_c_input = Input(shape=(hps.dec_rnn_size, ))
  output, last_h, last_c = model.decoder_lstm(
                        actual_input_x,
                        initial_state=[decoder_h_input, decoder_c_input])
  # [(1, 1, dec_rnn_size), (1, dec_rnn_size), (1, dec_rnn_size)]
  output = model.output_layer(output)
  # (1, 1, n_out)

  decoder_initial_model = Model(batch_z, initial_state)
  decoder_model = Model([decoder_input, batch_z,
                        decoder_h_input, decoder_c_input],
                        [output, last_h, last_c])

  prev_state = decoder_initial_model.predict(z)
  prev_h, prev_c = np.split(prev_state, 2, 1)
  # (1, dec_rnn_size), (1, dec_rnn_size)

  strokes = np.zeros((seq_len, 5), dtype=np.float32)
  greedy = False
  temp =  1.0

  for i in range(seq_len):
    decoder_output, next_h, next_c = decoder_model.predict(
                                          [prev_x, z, prev_h, prev_c])
    out = sketch_rnn_model.get_mixture_coef(decoder_output, model.n_out)
    [o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, o_pen, o_pen_logits] = out

    o_pi = K.eval(o_pi)
    o_mu1 = K.eval(o_mu1)
    o_mu2 = K.eval(o_mu2)
    o_sigma1 = K.eval(o_sigma1)
    o_sigma2 = K.eval(o_sigma2)
    o_corr = K.eval(o_corr)
    o_pen = K.eval(o_pen)

    if i < 0:
      greedy = False
      temp = 1.0
    else:
      greedy = greedy_mode
      temp = temperature

    idx = get_pi_idx(random.random(), o_pi[0], temp, greedy)

    idx_eos = get_pi_idx(random.random(), o_pen[0], temp, greedy)
    eos=[0, 0, 0]
    eos[idx_eos] = 1

    next_x1, next_x2 = sample_gaussian_2d(o_mu1[0][idx], o_mu2[0][idx],
                                          o_sigma1[0][idx], o_sigma2[0][idx],
                                          o_corr[0][idx], np.sqrt(temp), greedy)

    strokes[i, :] = [next_x1, next_x2, eos[0], eos[1], eos[2]]

    prev_x = np.zeros((1, 1, 5), dtype=np.float32)
    prev_x[0][0] = np.array(
        [next_x1, next_x2, eos[0], eos[1], eos[2]], dtype=np.float32)
    prev_h, prev_c = next_h, next_c

  # delete model to avoid a memory leak
  K.clear_session()

  return strokes


if __name__ == '__main__':
  train()
