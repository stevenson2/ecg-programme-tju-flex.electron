#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_dual_expert_triage.py — 双专家分诊 PC 模拟（A2）
=====================================================
按 dual_expert_deployment_plan.md A2：
  窗口 → 关卡（正常丢弃）→ 疑似异常 → 专家（P2A/exp7c 心律失常 + KD 心梗）→ OR
评估口径：患者级 seed=42 + deploy 链，MIT/INCART 与 PTB 双域。

产物：models/dual_expert_triage_eval.json
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import MODELS_DIR
import data.dataset as dataset
from data.patient_split import build_ptb_patient_map, patient_level_split

def load_model(rel):
    p = MODELS_DIR / rel
    print('load', p, flush=True)
    return tf.keras.models.load_model(str(p), compile=False)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--gate', default='gate/gate_model_resnet_large.h5')
    ap.add_argument('--arrhythmia', default='archived/final_resnet_l_p2a_backup.h5')
    ap.add_argument('--mi', default='final_kd_a070_t1.h5')
    ap.add_argument('--exp7c', default='best_resnet_large_exp7c.h5')
    ap.add_argument('--out', default='models/dual_expert_triage_eval.json')
    args=ap.parse_args()

    gate=load_model(args.gate)
    arr=load_model(args.arrhythmia)
    mi=load_model(args.mi)
    exp7c=load_model(args.exp7c)

    dataset.set_npz_suffix('_deploy')
    ds=dataset.prepare_datasets(batch_size=512, use_incart=True, use_ptb_beat=True,
                                ptb_abn_max=10000, domain_balanced=True, patient_split=True)
    x_test=ds['data']['test'][0].astype(np.float32)[..., np.newaxis]
    y_test=ds['data']['test'][1].astype(np.int32)
    ptb=dataset.load_ptb_data()
    _,_,te_m,_=patient_level_split(ptb['record_ids'], build_ptb_patient_map())
    x_ptb=ptb['beats'][te_m].astype(np.float32)[..., np.newaxis]
    y_ptb=ptb['labels'][te_m].astype(np.int32)

    def pred(m,x): return m.predict(x, batch_size=512, verbose=0)[:,1]
    p_gate_test=pred(gate,x_test); p_gate_ptb=pred(gate,x_ptb)
    p_arr_test=pred(arr,x_test); p_arr_ptb=pred(arr,x_ptb)
    p_mi_test=pred(mi,x_test); p_mi_ptb=pred(mi,x_ptb)
    p_exp_test=pred(exp7c,x_test); p_exp_ptb=pred(exp7c,x_ptb)

    gate_thrs=[0.3,0.4,0.5,0.6,0.7]
    exp_thrs=[0.3,0.4,0.5,0.6,0.7]
    results={}
    for gthr in gate_thrs:
        for ethr in exp_thrs:
            for arr_name, pa_t, pa_p in [('p2a',p_arr_test,p_arr_ptb),('exp7c',p_exp_test,p_exp_ptb)]:
                # OR: arrhythmia OR MI, after gate
                for domain, y, pg, pa, pm in [('mit_incart',y_test,p_gate_test,pa_t,p_mi_test),('ptb',y_ptb,p_gate_ptb,pa_p,p_mi_ptb)]:
                    passed = pg >= gthr
                    score = np.where(passed, np.maximum(pa, pm), 0.0)
                    pred_abn = score >= ethr
                    n_norm=int((y==0).sum()); n_abn=int((y==1).sum())
                    fp=int(((y==0)&pred_abn).sum()); tp=int(((y==1)&pred_abn).sum()); fn=n_abn-tp
                    sn=tp/max(n_abn,1); ea=fp/max(n_norm,1)
                    prec=tp/max(tp+fp,1)
                    key=(gthr,ethr,arr_name,domain)
                    results['_'.join(f'{k}' for k in key)]={'gate_theta':gthr,'expert_theta':ethr,'arrhythmia_expert':arr_name,'domain':domain,
                        'n':int(len(y)),'n_abn':n_abn,'n_normal':n_norm,'Sn_A':round(sn,4),'E_A':round(ea,4),
                        'precision':round(prec,4),'tp':int(tp),'fp':int(fp),'fn':int(fn)}
    out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps({'meta':{'task':'dual expert triage PC simulation A2','models':{'gate':args.gate,'arrhythmia':args.arrhythmia,'mi':args.mi,'exp7c':args.exp7c}},
        'results':results}, indent=2, ensure_ascii=False), encoding='utf-8')
    print('Saved',out)
    # print best E_A <=0.10 with max Sn combined? simple summary
    best=None
    for r in results.values():
        if r['domain']=='combined' and r['E_A']<=0.10 and r['Sn_A']>=0.6:
            if best is None or r['Sn_A']>best['Sn_A']: best=r
    print('best combined E_A<=0.10 Sn>=0.6:', best)

if __name__=='__main__':
    main()
