import os
import numpy as np
import random
import tensorflow as tf
import sketch_rnn_model


# A note on formats:
# Sketches are encoded as a sequence of strokes. stroke-3 and stroke-5 are
# different stroke encodings.
#   stroke-3 uses 3-tuples, consisting of x-offset, y-offset, and a binary
#       variable which is 1 if the pen is lifted between this position and
#       the next, and 0 otherwise.
#   stroke-5 consists of x-offset, y-offset, and p_1, p_2, p_3, a binary
#   one-hot vector of 3 possible pen states: pen down, pen up, end of sketch.
#   See section 3.1 of https://arxiv.org/abs/1704.03477 for more detail.
# Sketch-RNN takes input in stroke-5 format, with sketches padded to a common
# maximum length and prefixed by the special start token [0, 0, 1, 0, 0]
# The QuickDraw dataset is stored using stroke-3.
def get_bounds(data, factor=10):
  """Return bounds of data."""
  min_x = 0
  max_x = 0
  min_y = 0
  max_y = 0

  abs_x = 0
  abs_y = 0
  for i in range(len(data)):
    x = float(data[i, 0]) / factor
    y = float(data[i, 1]) / factor
    abs_x += x
    abs_y += y
    min_x = min(min_x, abs_x)
    min_y = min(min_y, abs_y)
    max_x = max(max_x, abs_x)
    max_y = max(max_y, abs_y)

  return (min_x, max_x, min_y, max_y)


def to_big_strokes(stroke, max_len=250):
  """Converts from stroke-3 to stroke-5 format and pads to given length."""
  # (But does not insert special start token).

  result = np.zeros((max_len, 5), dtype=float)
  l = len(stroke)
  assert l <= max_len
  result[0:l, 0:2] = stroke[:, 0:2]
  result[0:l, 3] = stroke[:, 2]
  result[0:l, 2] = 1 - result[0:l, 3]
  result[l:, 4] = 1
  return result


def to_normal_strokes(big_stroke):
  """Convert from stroke-5 format (from sketch-rnn paper) back to stroke-3."""
  l = 0
  for i in range(len(big_stroke)):
    if big_stroke[i, 4] > 0:
      l = i
      break
  if l == 0:
    l = len(big_stroke)
  result = np.zeros((l, 3))
  result[:, 0:2] = big_stroke[0:l, 0:2]
  result[:, 2] = big_stroke[0:l, 3]
  return result


def get_max_len(strokes):
  """Return the maximum length of an array of strokes."""
  max_len = 0
  for stroke in strokes:
    ml = len(stroke)
    if ml > max_len:
      max_len = ml
  return max_len


def slerp(p0, p1, t):
  """Spherical interpolation."""
  omega = np.arccos(np.dot(p0 / np.linalg.norm(p0), p1 / np.linalg.norm(p1)))
  so = np.sin(omega)
  return np.sin((1.0 - t) * omega) / so * p0 + np.sin(t * omega) / so * p1


def lerp(p0, p1, t):
  """Linear interpolation."""
  return (1.0 - t) * p0 + t * p1


def load_env(data_dir, model_dir):
  """Loads environment for inference mode, used in jupyter notebook."""
  model_params = sketch_rnn_model.get_default_hparams()
  with tf.gfile.Open(os.path.join(model_dir, 'model_config.json'), 'r') as f:
    model_params.parse_json(f.read())
  return load_dataset(data_dir, model_params)


class DataLoader(object):
  """Class for loading data."""

  def __init__(self,
               strokes,
               batch_size=100,
               max_seq_length=250,
               scale_factor=1.0,
               random_scale_factor=0.0,
               limit=1000):
    self.batch_size = batch_size  # minibatch size
    self.max_seq_length = max_seq_length  # N_max in sketch-rnn paper
    self.scale_factor = scale_factor  # divide offsets by this factor
    self.random_scale_factor = random_scale_factor  # data augmentation method
    # Removes large gaps in the data. x and y offsets are clamped to have
    # absolute value no greater than this limit.
    self.limit = limit
    self.start_stroke_token = [0, 0, 1, 0, 0]  # S_0 in sketch-rnn paper
    # sets self.strokes (list of ndarrays, one per sketch, in stroke-3 format,
    # sorted by size)
    self.preprocess(strokes)

  def preprocess(self, strokes):
    """Remove entries from strokes having > max_seq_length points."""
    raw_data = []
    seq_len = []
    count_data = 0

    for i in range(len(strokes)):
      data = strokes[i]
      if len(data) <= (self.max_seq_length):
        count_data += 1
        # removes large gaps from the data
        data = np.minimum(data, self.limit)
        data = np.maximum(data, -self.limit)
        data = np.array(data, dtype=np.float32)
        data[:, 0:2] /= self.scale_factor
        raw_data.append(data)
        seq_len.append(len(data))
    seq_len = np.array(seq_len)  # nstrokes for each sketch
    idx = np.argsort(seq_len)
    self.strokes = []
    for i in range(len(seq_len)):
      self.strokes.append(raw_data[idx[i]])
    print("total images <= max_seq_len is %d" % count_data)
    self.num_batches = int(count_data / self.batch_size)

  def random_sample(self):
    """Return a random sample, in stroke-3 format as used by draw_strokes."""
    sample = np.copy(random.choice(self.strokes))
    return sample

  def random_scale(self, data):
    """Augment data by stretching x and y axis randomly [1-e, 1+e]."""
    x_scale_factor = (
        np.random.random() - 0.5) * 2 * self.random_scale_factor + 1.0
    y_scale_factor = (
        np.random.random() - 0.5) * 2 * self.random_scale_factor + 1.0
    result = np.copy(data)
    result[:, 0] *= x_scale_factor
    result[:, 1] *= y_scale_factor
    return result

  def calculate_normalizing_scale_factor(self):
    """Calculate the normalizing factor explained in appendix of sketch-rnn."""
    data = []
    for i in range(len(self.strokes)):
      if len(self.strokes[i]) > self.max_seq_length:
        continue
      for j in range(len(self.strokes[i])):
        data.append(self.strokes[i][j, 0])
        data.append(self.strokes[i][j, 1])
    data = np.array(data)
    return np.std(data)

  def normalize(self, scale_factor=None):
    """Normalize entire dataset (delta_x, delta_y) by the scaling factor."""
    if scale_factor is None:
      scale_factor = self.calculate_normalizing_scale_factor()
    self.scale_factor = scale_factor
    for i in range(len(self.strokes)):
      self.strokes[i][:, 0:2] /= self.scale_factor

  def pad_strokes(self):
    """Pad the strokes to be stroke-5 bigger format as described in paper."""
    result = np.zeros(
                (len(self.strokes), self.max_seq_length + 1, 5), dtype=float)
    for i in range(len(self.strokes)):
      l = len(self.strokes[i])
      assert l <= self.max_seq_length
      result[i, 0:l, 0:2] = self.strokes[i][:, 0:2]
      result[i, 0:l, 3] = self.strokes[i][:, 2]
      result[i, 0:l, 2] = 1 - result[i, 0:l, 3]
      result[i, l:, 4] = 1
      # put in the first token, as described in sketch-rnn methodology
      result[i, 1:, :] = result[i, :-1, :]
      result[i, 0, :] = 0
      result[i, 0, 2] = self.start_stroke_token[2]  # setting S_0 from paper.
      result[i, 0, 3] = self.start_stroke_token[3]
      result[i, 0, 4] = self.start_stroke_token[4]
    return result


def load_dataset(data_dir, model_params):
  """Loads the .npz file, and splits the set into train/valid/test."""

  # normalizes the x and y columns usint the training set.
  # applies same scaling factor to valid and test set.

  datasets = []
  if isinstance(model_params.data_set, list):
    datasets = model_params.data_set
  else:
    datasets = [model_params.data_set]

  train_strokes = None
  valid_strokes = None
  test_strokes = None

  for dataset in datasets:
    data_filepath = os.path.join(data_dir, dataset)
    if data_dir.startswith('http://') or data_dir.startswith('https://'):
      tf.logging.info('Downloading %s', data_filepath)
      response = requests.get(data_filepath)
      data = np.load(BytesIO(response.content))
    else:
      data = np.load(data_filepath, encoding='latin1')
    tf.logging.info('Loaded {}/{}/{} from {}'.format(
        len(data['train']), len(data['valid']), len(data['test']), dataset))
    if train_strokes is None:
      train_strokes = data['train']
      valid_strokes = data['valid']
      test_strokes = data['test']
    else:
      train_strokes = np.concatenate((train_strokes, data['train']))
      valid_strokes = np.concatenate((valid_strokes, data['valid']))
      test_strokes = np.concatenate((test_strokes, data['test']))

  all_strokes = np.concatenate((train_strokes, valid_strokes, test_strokes))
  num_points = 0
  for stroke in all_strokes:
    num_points += len(stroke)
  avg_len = num_points / len(all_strokes)
  tf.logging.info('Dataset combined: {} ({}/{}/{}), avg len {}'.format(
      len(all_strokes), len(train_strokes), len(valid_strokes),
      len(test_strokes), int(avg_len)))

  # calculate the max strokes we need.
  max_seq_len = get_max_len(all_strokes)
  # overwrite the hps with this calculation.
  model_params.max_seq_len = max_seq_len

  tf.logging.info('model_params.max_seq_len %i.', model_params.max_seq_len)

  train_set = DataLoader(
      train_strokes,
      model_params.batch_size,
      max_seq_length=model_params.max_seq_len,
      random_scale_factor=model_params.random_scale_factor)

  normalizing_scale_factor = train_set.calculate_normalizing_scale_factor()
  train_set.normalize(normalizing_scale_factor)

  valid_set = DataLoader(
      valid_strokes,
      model_params.batch_size,
      max_seq_length=model_params.max_seq_len,
      random_scale_factor=0.0)
  valid_set.normalize(normalizing_scale_factor)

  test_set = DataLoader(
      test_strokes,
      model_params.batch_size,
      max_seq_length=model_params.max_seq_len,
      random_scale_factor=0.0)
  test_set.normalize(normalizing_scale_factor)

  tf.logging.info('normalizing_scale_factor %4.4f.', normalizing_scale_factor)

  result = [train_set, valid_set, test_set, model_params]
  return result
