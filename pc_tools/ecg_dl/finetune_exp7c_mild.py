#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""finetune_exp7c_mild.py — 温和版后训练：真实 AFE 正常拍 + 少量合成硬负样本
================================================================================
与 hardneg 版本对比：减少合成硬负样本数量、降低正常类权重，目标是在不大幅
损失 MIT/PTB AUC 的前提下降低真实 AFE 正常拍置信度。
"""
import sys, json, os, time
from pathlib import Path
import numpy as np
import tensorflow as tf
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import BEAT_WINDOW_SAMPLES
from models.resnet_lite_1d import build_ecg_resnet_lite_large

BASE = Path(__file__).resolve().parent
MODELS = BASE / "models"
DATA_REAL = BASE / "data" / "real"
CACHE = MODELS / "deploy_match"
ECG_DATA = Path(os.environ.get("ECG_PROCESSED_DIR", "/home/devcontainers/ecg_data"))
OUT_H5 = MODELS / "best_resnet_large_exp7c_v3.h5"
OUT_JSON = CACHE / "finetune_exp7c_v3.json"
OUT_CSV = MODELS / "train_history_exp7c_v3.csv"
SEED=42; rng=np.random.default_rng(SEED); tf.random.set_seed(SEED)

real1=np.load(DATA_REAL/'real_normal_beats_exp7c.npy').astype(np.float32)
real2=np.load(DATA_REAL/'real_normal_beats_rec_latest.npy').astype(np.float32)
real=np.concatenate([real1,real2],axis=0)
def load_domain(tag,n_abn,n_norm):
    b=np.load(ECG_DATA/f'{tag}_processed_deploy_causal_beats.npy',mmap_mode='r')
    l=np.load(ECG_DATA/f'{tag}_processed_deploy_causal_labels.npy',mmap_mode='r')
    ia=np.where(l==1)[0]; inn=np.where(l==0)[0]
    sa=rng.choice(ia,min(n_abn,len(ia)),replace=False); sn=rng.choice(inn,min(n_norm,len(inn)),replace=False)
    # 患者级泄漏守卫 (2026-09 审计: 本脚本历史抽样混入测试患者, 见
    # models/deploy_match/provenance_leakage_audit.json); 再次运行将直接失败。
    from pathlib import Path as _P
    import sys as _sys
    _sys.path.insert(0, str(_P(__file__).resolve().parent))
    from data.split_guard import get_guard
    import os as _os
    _ecg = _os.environ.get("ECG_PROCESSED_DIR", "/home/devcontainers/ecg_data")
    r = np.load(str(_ecg) + "/" + tag + "_processed_deploy_causal_record_ids.npy")
    get_guard(tag).assert_train_only(np.concatenate([r[sa], r[sn]]),
                                     context="load_domain")
    return (np.asarray(b[sa],dtype=np.float32),np.ones(len(sa),dtype=np.int32),
            np.asarray(b[sn],dtype=np.float32),np.zeros(len(sn),dtype=np.int32))
mit_a,mit_al,mit_n,mit_nl=load_domain('mit_bih',1200,400)
inc_a,inc_al,inc_n,inc_nl=load_domain('incart',300,100)
ptb_a,ptb_al,ptb_n,ptb_nl=load_domain('ptb',500,100)
x_mix=np.concatenate([mit_a,inc_a,ptb_a,mit_n,inc_n,ptb_n])[...,np.newaxis]
y_mix=np.concatenate([mit_al,inc_al,ptb_al,mit_nl,inc_nl,ptb_nl])
perm=rng.permutation(len(x_mix)); x_mix,y_mix=x_mix[perm],y_mix[perm]

# mild hard: only 20dB noise + baseline 0.3Hz + few impulses, 3 variants, not 10
hard=[]
for db in (20,):
    hard.append(real + rng.normal(0,10**(-db/20.0),real.shape).astype(np.float32))
t=np.arange(BEAT_WINDOW_SAMPLES,dtype=np.float32)/250.0
for freq,amp in ((0.3,0.3),(0.8,0.5)):
    hard.append(real + (amp*np.sin(2*np.pi*freq*t)).astype(np.float32)[None,:])
for n_imp in (2,5):
    y=real.copy()
    for _ in range(n_imp):
        pos=rng.integers(0,BEAT_WINDOW_SAMPLES); amp=float(rng.uniform(0.3,1.0)*rng.choice([-1,1])); y[:,pos]+=amp
    hard.append(y.astype(np.float32))
hard=np.concatenate(hard,axis=0)
print('hard mild count',len(hard))

val_idx=rng.choice(len(real),40,replace=False); trn_idx=np.setdiff1d(np.arange(len(real)),val_idx)
x_rtr=real[trn_idx][...,np.newaxis]; y_rtr=np.zeros(len(trn_idx),dtype=np.int32)
x_rva=real[val_idx][...,np.newaxis]; y_rva=np.zeros(len(val_idx),dtype=np.int32)
x_hard=hard[...,np.newaxis]; y_hard=np.zeros(len(hard),dtype=np.int32)
x_train=np.concatenate([x_mix,x_rtr,x_hard]); y_train=np.concatenate([y_mix,y_rtr,y_hard])
x_val=np.concatenate([x_mix[-400:],x_rva]); y_val=np.concatenate([y_mix[-400:],y_rva])
print('train',len(x_train),'val',len(x_val))

model=build_ecg_resnet_lite_large(input_shape=(BEAT_WINDOW_SAMPLES,1))
model.load_weights(str(MODELS/'best_resnet_large_exp7b.h5'))
for layer in model.layers: layer.trainable=layer.name in ('fc1','out')
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),loss=tf.keras.losses.SparseCategoricalCrossentropy(),metrics=['accuracy'])

def _auc(y,p):
    y=np.asarray(y).ravel(); p=np.asarray(p).ravel(); order=np.argsort(p); y=y[order]; n1=int((y==1).sum()); n0=int((y==0).sum())
    if n1==0 or n0==0: return 0.5
    ranks=np.arange(1,len(y)+1)[y==1].sum(); return float((ranks-n1*(n1+1)/2)/(n1*n0))
class CB(tf.keras.callbacks.Callback):
    def __init__(self,xv,yv): super().__init__(); self.xv=xv; self.yv=yv; self.best=0
    def on_epoch_end(self,epoch,logs=None):
        p=self.model.predict(self.xv,batch_size=128,verbose=0)[:,1]; auc=_auc(self.yv,p); logs=logs or {}; logs['val_auc']=auc
        if auc>self.best: self.best=auc; self.model.save(str(OUT_H5)); print(f'  * saved best {auc:.4f}')

m0=tf.keras.models.load_model(str(MODELS/'best_resnet_large_exp7b.h5'),compile=False)
p0=m0.predict(real[...,np.newaxis],batch_size=64,verbose=0)[:,1]
print('BEFORE real mean',p0.mean())
cbs=[CB(x_val,y_val),tf.keras.callbacks.EarlyStopping(monitor='val_auc',mode='max',patience=12,restore_best_weights=True,verbose=1),tf.keras.callbacks.CSVLogger(str(OUT_CSV))]
t0=time.time(); hist=model.fit(x_train,y_train,validation_data=(x_val,y_val),batch_size=32,epochs=40,callbacks=cbs,class_weight={0:2.0,1:1.0},verbose=2)
print('done',time.time()-t0)
model.load_weights(str(OUT_H5)); tf.keras.backend.clear_session(); model=tf.keras.models.load_model(str(OUT_H5),compile=False)
p1=model.predict(real[...,np.newaxis],batch_size=64,verbose=0)[:,1]
p1h=model.predict(hard[...,np.newaxis],batch_size=64,verbose=0)[:,1]
p1v=model.predict(x_rva,batch_size=64,verbose=0)[:,1]
print('AFTER real mean',p1.mean(),'frac>0.5',(p1>0.5).mean())
print('AFTER hard mean',p1h.mean(),'frac>0.5',(p1h>0.5).mean())
print('heldout',p1v.mean())
out={'date':time.strftime('%Y-%m-%d %H:%M:%S'),'purpose':'mild exp7c_v3','data':{'real':len(real),'hard':len(hard),'mix_abn':int((y_mix==1).sum()),'mix_norm':int((y_mix==0).sum()),'heldout':len(val_idx)},'config':{'class_weight':{0:2.0,1:1.0},'epochs':int(len(hist.epoch))},'confidence':{'before':{'mean':float(p0.mean()),'frac_gt_0.5':float((p0>0.5).mean())},'after_v3':{'mean':float(p1.mean()),'frac_gt_0.5':float((p1>0.5).mean())},'hard':{'mean':float(p1h.mean()),'frac_gt_0.5':float((p1h>0.5).mean())},'heldout40':{'mean':float(p1v.mean())}}}
CACHE.mkdir(parents=True,exist_ok=True); OUT_JSON.write_text(json.dumps(out,indent=2,ensure_ascii=False))
print('saved',OUT_H5)
