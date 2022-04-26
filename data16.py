from keras.layers import Embedding
from keras import Sequential
from collections import Counter
from keras.layers import SimpleRNN
from keras.layers import Dense
from keras.layers import Bidirectional
from keras.layers import LSTM
from keras.layers import GRU
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import pandas as pd

tf.random.set_seed(1)
rnn_layer = tf.keras.layers.SimpleRNN(
    units=2, use_bias=True, return_sequences=True)
rnn_layer.build(input_shape=(None, None, 5))
w_xh, w_oo, b_h = rnn_layer.weights
print('W_xh shape:', w_xh.shape)
print('W_oo shape:', w_oo.shape)
print('b_h shape:', b_h.shape)

x_seq = tf.convert_to_tensor([[1.0]*5, [2.0]*5, [3.0]*5], dtype=tf.float32)
# SimpleRNNの出力
output = rnn_layer(tf.reshape(x_seq, shape=(1, 3, 5)))
# 出力を手動で計算
out_man = []
for t in range(len(x_seq)):
    xt = tf.reshape(x_seq[t], (1, 5))
    print('Time step {} =>'.format(t))
    print('    Input          :', xt.numpy())

    ht = tf.matmul(xt, w_xh) + b_h
    print('    Output          :', ht.numpy())

    if t > 0:
        prev_o = out_man[t-1]
    else:
        prev_o = tf.zeros(shape=(ht.shape))
    ot = ht+tf.matmul(prev_o, w_oo)
    ot = tf.math.tanh(ot)
    out_man.append(ot)
    print('    Output (manul) :', ot.numpy())
    print('    SimpleRNN output:'.format(t), output[0][t].numpy())

df = pd.read_csv('movie_data.csv', encoding='utf-8')

# 手順1:Datasetを作成
target = df.pop('sentiment')
ds_raw = tf.data.Dataset.from_tensor_slices((df.values, target.values))
# 調査
for ex in ds_raw.take(3):
    tf.print(ex[0].numpy()[0][:50], ex[1])

tf.random.set_seed(1)
ds_raw = ds_raw.shuffle(50000, reshuffle_each_iteration=False)
ds_raw_test = ds_raw.take(25000)
ds_raw_train_valid = ds_raw.skip(25000)
ds_raw_train = ds_raw_train_valid.take(20000)
ds_raw_valid = ds_raw_train_valid.skip(20000)

# 手順2:一意な単語を特定
try:
    tokenizer = tfds.features.text.Tokenizer()
except AttributeError:
    tokenizer = tfds.deprecated.text.Tokenizer()
token_counts = Counter()

for example in ds_raw_train:
    tokens = tokenizer.tokenize(example[0].numpy()[0])
    token_counts.update(tokens)

print('Vocab-size:', len(token_counts))

# 手順3
try:
    encoder = tfds.features.text.TokenTextEncoder(token_counts)
except AttributeError:
    encoder = tfds.deprecated.text.TokenTextEncoder(token_counts)

example_str = 'This is an example!'
print(encoder.encode(example_str))

# 手順3:変換用の関数を定義


def encode(text_tensor, label):
    text = text_tensor.numpy()[0]
    encoded_text = encoder.encode(text)
    return encoded_text, label


def encode_map_fn(text, label):
    return tf.py_function(encode, inp=[text, label], Tout=(tf.int64, tf.int64))


ds_train = ds_raw_train.map(encode_map_fn)
ds_valid = ds_raw_valid.map(encode_map_fn)
ds_test = ds_raw_test.map(encode_map_fn)
# 訓練データの形状をチェック
tf.random.set_seed(1)
for example in ds_train.shuffle(1000).take(5):
    print('Sequence length:', example[0].shape)

# 小さなサブセットを取得
ds_subset = ds_train.take(8)
for example in ds_subset:
    print('Individual size:', example[0].shape)

# このサブセットをバッチに分割
ds_batched = ds_subset.padded_batch(4, padded_shapes=([-1], []))
for batch in ds_batched:
    print('Batch dimension:', batch[0].shape)

train_data = ds_train.padded_batch(32, padded_shapes=([-1], []))
valid_data = ds_valid.padded_batch(32, padded_shapes=([-1], []))
test_data = ds_test.padded_batch(32, padded_shapes=([-1], []))

model = Sequential()
model.add(Embedding(input_dim=1000, output_dim=32))
model.add(SimpleRNN(32, return_sequences=True))
model.add(SimpleRNN(32))
model.add(Dense(1))
model.summary()

embedding_dim = 20
vocab_size = len(token_counts)+2
tf.random.set_seed(1)
# モデルを構築
bi_lstm_model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim,
                              name='embed-layer'),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, name='lstm-layer'),
                                  name='bidir-lste'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
bi_lstm_model.summary()
# コンパイルと訓練
bi_lstm_model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                      loss=tf.keras.losses.BinaryCrossentropy(
                          from_logits=False),
                      metrics=['accuracy'])
history = bi_lstm_model.fit(train_data, validation_data=valid_data, epochs=10)

# テストデータでの評価
test_results = bi_lstm_model.evaluate(test_data)
print('Test Acc.: {:.2f}%'.format(test_results[1]*100))


def preprocess_datasets(
        ds_raw_train,
        ds_raw_valid,
        ds_raw_test,
        max_seq_length=None,
        batch_size=32):

    # Step 1: (already done => creating a dataset)
    # Step 2: find unique tokens

    try:
        tokenizer = tfds.features.text.Tokenizer()
    except AttributeError:
        tokenizer = tfds.deprecated.text.Tokenizer()

    token_counts = Counter()

    for example in ds_raw_train:
        tokens = tokenizer.tokenize(example[0].numpy()[0])
        if max_seq_length is not None:
            tokens = tokens[-max_seq_length:]
        token_counts.update(tokens)

    print('Vocab-size:', len(token_counts))

    # Step 3: encoding the texts
    try:
        encoder = tfds.features.text.TokenTextEncoder(token_counts)
    except AttributeError:
        encoder = tfds.deprecated.text.TokenTextEncoder(token_counts)

    def encode(text_tensor, label):
        text = text_tensor.numpy()[0]
        encoded_text = encoder.encode(text)
        if max_seq_length is not None:
            encoded_text = encoded_text[-max_seq_length:]
        return encoded_text, label

    def encode_map_fn(text, label):
        return tf.py_function(encode, inp=[text, label],
                              Tout=(tf.int64, tf.int64))

    ds_train = ds_raw_train.map(encode_map_fn)
    ds_valid = ds_raw_valid.map(encode_map_fn)
    ds_test = ds_raw_test.map(encode_map_fn)

    # Step 4: batching the datasets
    train_data = ds_train.padded_batch(
        batch_size, padded_shapes=([-1], []))

    valid_data = ds_valid.padded_batch(
        batch_size, padded_shapes=([-1], []))

    test_data = ds_test.padded_batch(
        batch_size, padded_shapes=([-1], []))

    return (train_data, valid_data,
            test_data, len(token_counts))


def build_rnn_model(embedding_dim, vocab_size,
                    recurrent_type='SimpleRNN',
                    n_recurrent_units=64,
                    n_recurrent_layers=1,
                    bidirectional=True):

    tf.random.set_seed(1)

    # build the model
    model = tf.keras.Sequential()

    model.add(
        Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            name='embed-layer')
    )

    for i in range(n_recurrent_layers):
        return_sequences = (i < n_recurrent_layers-1)

        if recurrent_type == 'SimpleRNN':
            recurrent_layer = SimpleRNN(
                units=n_recurrent_units,
                return_sequences=return_sequences,
                name='simprnn-layer-{}'.format(i))
        elif recurrent_type == 'LSTM':
            recurrent_layer = LSTM(
                units=n_recurrent_units,
                return_sequences=return_sequences,
                name='lstm-layer-{}'.format(i))
        elif recurrent_type == 'GRU':
            recurrent_layer = GRU(
                units=n_recurrent_units,
                return_sequences=return_sequences,
                name='gru-layer-{}'.format(i))

        if bidirectional:
            recurrent_layer = Bidirectional(
                recurrent_layer, name='bidir-'+recurrent_layer.name)

        model.add(recurrent_layer)

    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    return model


batch_size = 32
embedding_dim = 20
max_seq_length = 100
train_data, valid_data, test_data, n = preprocess_datasets(
    ds_raw_train, ds_raw_valid, ds_raw_test,
    max_seq_length=max_seq_length, batch_size=batch_size
)
vocab_size = n+2
rnn_model = build_rnn_model(embedding_dim, vocab_size, recurrent_type='SimpleRNN',
                            n_recurrent_units=64, n_recurrent_layers=1,
                            bidirectional=True)
rnn_model.summary()

rnn_model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                  metrics=['accuracy'])
history = rnn_model.fit(
    train_data,
    validation_data=valid_data,
    epochs=10)

results = rnn_model.evaluate(test_data)
print('Test Acc.: {:.2f}%'.format(results[1]*100))
