#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_gate_model.py — A1 关卡验收（患者级、部署链）
====================================================
评估 gate_model_*.h5 在 MIT+INCART 与 PTB 测试集上的：
  Sn_A = 异常保留率（Abnormal sensitivity）
  E_A  = 正常误放行率/误报率（Normal false-alarm rate）
并扫描阈值，输出可满足 Sn_A>=95% 时最低 E_A 的操作点。
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, tensorflow as tf
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import MODELS_DIR
import data.dataset as dataset
from data.patient_split import build_ptb_patient_map, patient_level_split
from eval_aami_matrix import add_channel_dim

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='gate/gate_model_resnet_medium.h5')
    ap.add_argument('--out', default='models/gate/gate_model_eval.json')
    args=ap.parse_args()
    model_path = MODELS_DIR / args.model
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    model=tf.keras.models.load_model(str(model_path), compile=False)
    dataset.set_npz_suffix('_deploy')

    ds=dataset.prepare_datasets(batch_size=512, use_incart=True, use_ptb_beat=True,
                                ptb_abn_max=10000, domain_balanced=True, patient_split=True)
    x_test=ds['data']['test'][0].astype(np.float32)[..., np.newaxis]
    y_test=ds['data']['test'][1].astype(np.int32)
    p_test=model.predict(x_test, batch_size=512, verbose=0)[:,1]

    ptb=dataset.load_ptb_data()
    _,_,te_m,_=patient_level_split(ptb['record_ids'], build_ptb_patient_map())
    x_ptb=ptb['beats'][te_m].astype(np.float32)[..., np.newaxis]
    y_ptb=ptb['labels'][te_m].astype(np.int32)
    p_ptb=model.predict(x_ptb, batch_size=512, verbose=0)[:,1]

    results={}
    for name,y,p in [('mit_incart',y_test,p_test),('ptb',y_ptb,p_ptb),('combined',np.concatenate([y_test,y_ptb]),np.concatenate([p_test,p_ptb]))]:
        n=len(y); abn=int(y.sum()); norm=n-abn
        auc=float(roc_auc_score(y,p))
        best=None
        for thr in np.arange(0.1,0.95,0.01):
            sn=float(((y==1)&(p>=thr)).sum()/max(abn,1))
            ea=float(((y==0)&(p>=thr)).sum()/max(norm,1))
            if sn>=0.95:
                if best is None or ea<best['E_A']:
                    best={'theta':round(float(thr),3),'Sn_A':round(sn,4),'E_A':round(ea,4)}
        # 若没有阈值满足 Sn>=95，给出 max Sn 且 E_A 最低
        if best is None:
            cand=[]
            for thr in np.arange(0.1,0.95,0.01):
                sn=float(((y==1)&(p>=thr)).sum()/max(abn,1))
                ea=float(((y==0)&(p>=thr)).sum()/max(norm,1))
                cand.append((sn,ea,thr))
            sn_max=max(c[0] for c in cand)
            best={'theta':None,'Sn_A':round(sn_max,4),'E_A':round(min(c[1] for c in cand if abs(c[0]-sn_max)<1e-9),4),'note':'no threshold meets Sn>=0.95'}
        results[name]={'n':n,'n_abn':abn,'n_normal':norm,'auc':round(auc,4),
                       'best_Sn>=95':best,
                       'thr_0.5':{'Sn_A':round(float(((y==1)&(p>=0.5)).sum()/max(abn,1)),4),
                                   'E_A':round(float(((y==0)&(p>=0.5)).sum()/max(norm,1)),4)}}
        print(name, results[name])

    out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(results,indent=2,ensure_ascii=False),encoding='utf-8')
    print('Saved',out)

if __name__=='__main__':
    main()
