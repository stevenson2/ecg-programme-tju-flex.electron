"""Deployed architecture (models.build('S'), 62,834 params) with two optional extensions:
ctx_dim>0 adds a scalar-context branch (Dense16) concatenated before fc1; n_out=4 gives an N/S/V/X head."""
import tensorflow as tf
from tensorflow.keras import layers
import source_model as source


def build_variant(seed, n_out=2, ctx_dim=0, aux4=False):
    tf.keras.utils.set_random_seed(seed)
    inp = layers.Input((250, 1), name='ecg_input')
    x = layers.Conv1D(16, 7, strides=2, padding='same', use_bias=False, name='stem')(inp)
    x = layers.BatchNormalization(name='stem_bn')(x)
    x = layers.ReLU(name='stem_rl')(x)
    bid = 0
    for f, n, k, s in zip((16, 32, 64, 128), (2, 3, 3, 1), (7, 5, 3, 3), (1, 2, 2, 1)):
        for bi in range(n):
            x = source.res_block(x, f, k, s if bi == 0 else 1, bid)
            bid += 1
    x = layers.GlobalAveragePooling1D(name='gap')(x)
    ins = [inp]
    if ctx_dim:
        c = layers.Input((ctx_dim,), name='ctx_input')
        h = layers.Dense(16, activation='relu', name='ctx_fc')(c)
        x = layers.Concatenate(name='cat')([x, h])
        ins.append(c)
    x = layers.Dense(64, activation='relu', name='fc1')(x)
    x = layers.Dropout(.4, seed=30000 + seed, name='do')(x)
    out = layers.Dense(n_out, activation='softmax', name='out')(x)
    if aux4:  # auxiliary N/S/V/X head; 'out' stays the deployable binary head
        out = [out, layers.Dense(4, activation='softmax', name='out4')(x)]
    return tf.keras.Model(ins if ctx_dim else inp, out, name='ecg_resnet_lite')
