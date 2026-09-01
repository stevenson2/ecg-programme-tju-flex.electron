#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_gate_model_3class.py — 三分类关卡训练（A1 备选）
======================================================
类别：
  0 = Normal
  1 = Arrhythmia suspected（MIT/INCART 异常）
  2 = MI suspected（PTB 异常）

数据：MIT+INCART deploy + PTB deploy，患者级 seed=42。
训练：三类按 max-per-class 下采样，保持类别均衡。
产物：models/gate/gate_model_3class_<arch>.h5 + history + meta
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import MODELS_DIR
import data.dataset as dataset
from data.patient_split import (
    build_mit_patient_map, build_incart_patient_map,
    build_ptb_patient_map, patient_level_split,
)
from models.resnet_lite_1d import build_ecg_resnet_lite, model_summary_table

_gpus=tf.config.list_physical_devices('GPU')
for _g in _gpus:
    try: tf.config.experimental.set_memory_growth(_g, True)
    except Exception: pass

ARCH_CFG={
 'resnet_large': dict(filters=(16,32,64,128), blocks_per_stage=(2,3,3,1),
                      kernel_sizes=(7,5,3,3), strides=(1,2,2,1), dropout_rate=0.4),
 'resnet_medium': dict(filters=(16,32,64,128), blocks_per_stage=(2,2,2,1),
                       kernel_sizes=(5,5,3,3), strides=(1,2,2,1), dropout_rate=0.3),
}

def onehot(y, n=3):
    return tf.keras.utils.to_categorical(y, num_classes=n)

def build_tf_dataset(x, y, batch_size, shuffle=True, repeat=False, steps=None):
    x=x.astype(np.float32)
    yy=onehot(y.astype(np.int32),3).astype(np.float32)
    ds=tf.data.Dataset.from_tensor_slices((x[...,np.newaxis], yy))
    if shuffle:
        ds=ds.shuffle(min(10000,len(x)))
    if repeat:
        ds=ds.repeat()
    ds=ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    if repeat and steps:
        ds=ds.take(steps)
    return ds

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--arch',choices=sorted(ARCH_CFG.keys()),default='resnet_large')
    ap.add_argument('--epochs',type=int,default=80)
    ap.add_argument('--patience',type=int,default=25)
    ap.add_argument('--batch-size',type=int,default=256)
    ap.add_argument('--steps-per-epoch',type=int,default=1000)
    ap.add_argument('--optimizer',choices=['adamw','sgd'],default='adamw')
    ap.add_argument('--lr',type=float,default=5e-4)
    ap.add_argument('--max-per-class',type=int,default=80000)
    ap.add_argument('--quick',action='store_true')
    ap.add_argument('--out-dir',default=str(MODELS_DIR/'gate'))
    args=ap.parse_args()
    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    dataset.set_npz_suffix('_deploy')

    print('Loading A (MIT+INCART) ...',flush=True)
    a=dataset.load_mit_incart_merged()
    pmap={}; pmap.update(build_mit_patient_map())
    pmap.update({rid+100000:'inc_'+pat for rid,pat in build_incart_patient_map().items()})
    tr_m,va_m,te_m,pstats=patient_level_split(a['record_ids'],pmap)
    a_tr_x,a_tr_y=a['beats'][tr_m],a['labels'][tr_m]
    a_va_x,a_va_y=a['beats'][va_m],a['labels'][va_m]
    a_te_x,a_te_y=a['beats'][te_m],a['labels'][te_m]
    # class1 = A abnormal
    a_tr_y = np.where(a_tr_y==1, 1, 0).astype(np.int32)
    a_va_y = np.where(a_va_y==1, 1, 0).astype(np.int32)
    a_te_y = np.where(a_te_y==1, 1, 0).astype(np.int32)

    print('Loading PTB ...',flush=True)
    ptb=dataset.load_ptb_data()
    trp,vap,tep,_=patient_level_split(ptb['record_ids'], build_ptb_patient_map())
    ptb_tr_x,ptb_tr_y=ptb['beats'][trp],ptb['labels'][trp]
    ptb_va_x,ptb_va_y=ptb['beats'][vap],ptb['labels'][vap]
    ptb_te_x,ptb_te_y=ptb['beats'][tep],ptb['labels'][tep]
    # class2 = PTB abnormal; PTB normal -> class0
    ptb_tr_y=np.where(ptb_tr_y==1,2,0).astype(np.int32)
    ptb_va_y=np.where(ptb_va_y==1,2,0).astype(np.int32)
    ptb_te_y=np.where(ptb_te_y==1,2,0).astype(np.int32)

    # Validation/test combined
    val_x=np.concatenate([a_va_x,ptb_va_x]); val_y=np.concatenate([a_va_y,ptb_va_y])
    te_x=np.concatenate([a_te_x,ptb_te_x]); te_y=np.concatenate([a_te_y,ptb_te_y])

    # Train balanced downsample per class
    rng=np.random.default_rng(42)
    tr_x_all=np.concatenate([a_tr_x,ptb_tr_x]); tr_y_all=np.concatenate([a_tr_y,ptb_tr_y])
    keep=[]
    for c in [0,1,2]:
        idx=np.where(tr_y_all==c)[0]
        if len(idx)>args.max_per_class:
            idx=rng.choice(idx,args.max_per_class,replace=False)
        keep.append(idx)
    keep=np.concatenate(keep)
    perm=rng.permutation(len(keep))
    tr_x,tr_y=tr_x_all[keep[perm]],tr_y_all[keep[perm]]
    print(f'Train counts: {np.bincount(tr_y)}',flush=True)
    print(f'Val counts: {np.bincount(val_y)}',flush=True)
    print(f'Test counts: {np.bincount(te_y)}',flush=True)

    train_ds=build_tf_dataset(tr_x,tr_y,args.batch_size,shuffle=True,repeat=True,steps=args.steps_per_epoch)
    val_ds=build_tf_dataset(val_x,val_y,args.batch_size,shuffle=False)
    if args.quick:
        train_ds=build_tf_dataset(tr_x,tr_y,args.batch_size,shuffle=True,repeat=True,steps=2)
        val_ds=val_ds.take(2)

    cfg=ARCH_CFG[args.arch]
    model=build_ecg_resnet_lite(input_shape=(250,1),n_classes=3,**cfg)
    model_summary_table(model)
    if args.optimizer=='adamw':
        opt=tf.keras.optimizers.AdamW(learning_rate=args.lr,weight_decay=1e-4)
    else:
        opt=tf.keras.optimizers.SGD(learning_rate=args.lr,momentum=0.9,nesterov=True,weight_decay=1e-4)
    model.compile(optimizer=opt,loss='categorical_crossentropy',
                  metrics=['accuracy',tf.keras.metrics.AUC(name='auc',multi_label=False)])
    model_path=out/f'gate_model_3class_{args.arch}.h5'
    history_csv=out/f'gate_model_3class_{args.arch}_history.csv'
    callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=args.patience,restore_best_weights=True,verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.5,patience=max(6,args.patience//3),min_lr=1e-6,verbose=1),
        tf.keras.callbacks.ModelCheckpoint(filepath=str(model_path),monitor='val_loss',mode='min',save_best_only=True,verbose=1),
        tf.keras.callbacks.CSVLogger(str(history_csv),append=False),
    ]
    t0=time.time()
    hist=model.fit(train_ds,validation_data=val_ds,
                   epochs=2 if args.quick else args.epochs,
                   steps_per_epoch=2 if args.quick else None,
                   validation_steps=2 if args.quick else None,
                   callbacks=callbacks,verbose=2)
    model.save(model_path)
    meta={'task':'three_class_gate_A1','arch':args.arch,'epochs_run':len(hist.history.get('loss',[])),
          'best_val_loss':float(min(hist.history.get('val_loss',[0]))),
          'best_val_accuracy':float(max(hist.history.get('val_accuracy',[0]))),
          'optimizer':args.optimizer,'lr':args.lr,'batch_size':args.batch_size,
          'max_per_class':args.max_per_class,'model_path':str(model_path),
          'history_csv':str(history_csv),'elapsed_s':round(time.time()-t0,1)}
    (out/f'gate_model_3class_{args.arch}_meta.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding='utf-8')
    print('[3class gate] done',meta,flush=True)

if __name__=='__main__':
    main()
