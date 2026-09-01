#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""eval_gate_model_3class.py — 三分类关卡测试混淆矩阵"""
import json,sys
from pathlib import Path
import numpy as np, tensorflow as tf
sys.path.insert(0,str(Path(__file__).resolve().parent))
from config import MODELS_DIR
import data.dataset as dataset
from data.patient_split import build_ptb_patient_map, patient_level_split

def main():
    model_path=MODELS_DIR/'gate/gate_model_3class_resnet_large.h5'
    m=tf.keras.models.load_model(str(model_path),compile=False)
    dataset.set_npz_suffix('_deploy')
    ds=dataset.prepare_datasets(batch_size=512,use_incart=True,use_ptb_beat=True,
                                ptb_abn_max=10000,domain_balanced=True,patient_split=True)
    xa=ds['data']['test'][0].astype(np.float32)[...,np.newaxis]
    ya=np.where(ds['data']['test'][1].astype(np.int32)==1,1,0)
    ptb=dataset.load_ptb_data()
    _,_,te,_=patient_level_split(ptb['record_ids'],build_ptb_patient_map())
    xp=ptb['beats'][te].astype(np.float32)[...,np.newaxis]
    yp=np.where(ptb['labels'][te].astype(np.int32)==1,2,0)
    x=np.concatenate([xa,xp]); y=np.concatenate([ya,yp])
    p=m.predict(x,batch_size=512,verbose=0)
    pred=np.argmax(p,axis=1)
    # threshold scan on normal probability
    cm=np.zeros((3,3),dtype=int)
    for t,pr in zip(y,pred): cm[t,pr]+=1
    print('confusion (rows true N/ARR/MI, cols pred):')
    print(cm)
    norm=cm[0].sum(); arr=cm[1].sum(); mi=cm[2].sum()
    out={'n_normal':int(norm),'n_arr':int(arr),'n_mi':int(mi),
         'normal_E_A':round(float(cm[0,1]+cm[0,2])/max(norm,1),4),
         'arrhythmia_recall':round(float(cm[1,1])/max(arr,1),4),
         'mi_recall':round(float(cm[2,2])/max(mi,1),4),
         'confusion':cm.tolist()}
    print(out)
    scan=[]
    for t in [x/100 for x in range(20,91,5)]:
        norm_ok = p[:,0]>=t
        pred2 = np.where(norm_ok, 0, np.argmax(p[:,1:],axis=1)+1)
        cm=np.zeros((3,3),dtype=int)
        for yy,pr in zip(y,pred2): cm[yy,pr]+=1
        ea=float(cm[0,1]+cm[0,2])/max(cm[0].sum(),1)
        ar=float(cm[1,1])/max(cm[1].sum(),1)
        mi=float(cm[2,2])/max(cm[2].sum(),1)
        scan.append({'t_norm':round(t,2),'E_A':round(ea,4),'arr_recall':round(ar,4),'mi_recall':round(mi,4),'confusion':cm.tolist()})
    print('\nscan: E_A<=0.05/0.10/0.15 best arr+mi')
    for lim in [0.05,0.10,0.15]:
        cand=[r for r in scan if r['E_A']<=lim]
        if cand:
            best=max(cand,key=lambda r:r['arr_recall']+r['mi_recall'])
            print(lim,best)
    out['threshold_scan']=scan
    Path('models/gate/gate_model_3class_eval.json').write_text(json.dumps(out,indent=2,ensure_ascii=False),encoding='utf-8')

if __name__=='__main__':
    main()
