import tensorflow as tf
from tensorflow.keras import layers
import source_model as source

@tf.keras.utils.register_keras_serializable(package='NVLongrun')
class FixedLowpass(layers.Layer):
    def __init__(self,bypass=False,**kwargs):
        super().__init__(**kwargs); self.bypass=bypass
    def call(self,x):
        if self.bypass: return tf.identity(x)
        z=tf.pad(x,[[0,0],[1,1],[0,0]])
        return .25*z[:,:-2,:]+.5*z[:,1:-1,:]+.25*z[:,2:,:]
    def get_config(self): return {**super().get_config(),'bypass':self.bypass}

def build(kind,seed,bypass=False):
    tf.keras.utils.set_random_seed(seed)
    filters=(64,128,256,512) if kind in ('L','L_LP') else (16,32,64,128)
    inp=layers.Input((250,1),name='ecg_input')
    x=FixedLowpass(bypass=bypass,name='lp_stem')(inp) if kind in ('S_LP','L_LP') else inp
    x=layers.Conv1D(16,7,strides=2,padding='same',use_bias=False,name='stem')(x)
    x=layers.BatchNormalization(name='stem_bn')(x); x=layers.ReLU(name='stem_rl')(x)
    bid=0
    for si,(f,n,k,s) in enumerate(zip(filters,(2,3,3,1),(7,5,3,3),(1,2,2,1))):
        for bi in range(n):
            stride=s if bi==0 else 1
            if kind in ('S_LP','L_LP') and bi==0 and si in (1,2): x=FixedLowpass(bypass=bypass,name=f'lp_b{bid}')(x)
            x=source.res_block(x,f,k,stride,bid); bid+=1
    x=layers.GlobalAveragePooling1D(name='gap')(x)
    x=layers.Dense(64,activation='relu',name='fc1')(x)
    x=layers.Dropout(.4,seed=30000+seed,name='do')(x)
    out=layers.Dense(2,activation='softmax',name='out')(x)
    model=tf.keras.Model(inp,out,name='ecg_resnet_lite')
    assert model.count_params()==(828514 if kind in ('L','L_LP') else 62834)
    assert model.output_shape==(None,2)
    return model

def copy_learned(src,dst):
    matches=[]
    for l in src.layers:
        if l.weights:
            other=dst.get_layer(l.name); other.set_weights(l.get_weights())
            import numpy as np
            assert all(np.array_equal(a,b) for a,b in zip(l.get_weights(),other.get_weights()))
            matches.append(l.name)
    return matches
