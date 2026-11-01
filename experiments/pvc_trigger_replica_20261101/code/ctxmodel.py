"""models.build('S') plus a 4-scalar context input (RR ratios, amplitude ratio) joined after GAP."""
import tensorflow as tf
from tensorflow.keras import layers
import source_model as source


def build_ctx(seed):
    tf.keras.utils.set_random_seed(seed)
    inp = layers.Input((250, 1), name='ecg_input')
    cin = layers.Input((4,), name='ctx_input')
    x = layers.Conv1D(16, 7, strides=2, padding='same', use_bias=False, name='stem')(inp)
    x = layers.BatchNormalization(name='stem_bn')(x); x = layers.ReLU(name='stem_rl')(x)
    bid = 0
    for f, n, k, s in zip((16, 32, 64, 128), (2, 3, 3, 1), (7, 5, 3, 3), (1, 2, 2, 1)):
        for bi in range(n):
            x = source.res_block(x, f, k, s if bi == 0 else 1, bid); bid += 1
    x = layers.GlobalAveragePooling1D(name='gap')(x)
    c = layers.Dense(16, activation='relu', name='ctx_fc')(cin)
    x = layers.Concatenate(name='join')([x, c])
    x = layers.Dense(64, activation='relu', name='fc1')(x)
    x = layers.Dropout(.4, seed=30000 + seed, name='do')(x)
    out = layers.Dense(2, activation='softmax', name='out')(x)
    return tf.keras.Model([inp, cin], out, name='ecg_resnet_lite_ctx')
