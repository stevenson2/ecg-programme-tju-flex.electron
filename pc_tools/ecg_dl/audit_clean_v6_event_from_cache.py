#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""audit_clean_v6_event_from_cache.py — 从缓存概率补全 v6 验证/测试事件输出
"""
import sys, json, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import MODELS_DIR
from data.dataset import set_npz_suffix, load_mit_incart_merged
from data.patient_split import build_mit_patient_map, build_incart_patient_map, patient_level_split
from eval_aami_breakdown import recover_mit_symbols_per_record, recover_incart_symbols_per_record, align_symbols_to_npz
from eval_exp7c_policy_sweep import reduce_mit_augmentation, evaluate_sequence_set, DEFAULT_GT_GAP
import eval_exp7c_policy_sweep as pol
OUT = Path(__file__).resolve().parent / "models" / "deploy_match" / "event_clean_v6_val_test.json"
CACHE = Path(__file__).resolve().parent / "models" / "deploy_match"
THETAS=[0.70,0.75,0.80,0.82,0.84,0.85,0.86,0.88,0.90]
COOLDOWNS=[5,6,8,10]

def main():
    t0=time.time()
    set_npz_suffix("_deploy_causal")
    data=load_mit_incart_merged()
    beats,labels,rids=data['beats'],data['labels'],data['record_ids']
    beats,labels,rids,kept=reduce_mit_augmentation(beats,labels,rids)
    per_rec_syms=recover_mit_symbols_per_record()
    incart_dir=Path(__file__).resolve().parent/'data'/'raw'/'incart'
    per_rec_syms.update(recover_incart_symbols_per_record(incart_dir))
    sym_full,_=align_symbols_to_npz(per_rec_syms,data['record_ids'],6)
    symbols=sym_full[kept]
    pmap={}; pmap.update(build_mit_patient_map())
    pmap.update({rid+100000:'inc_'+pat for rid,pat in build_incart_patient_map().items()})
    tr_m,va_m,te_m,pstats=patient_level_split(data['record_ids'],pmap)
    va_red,te_red=va_m[kept],te_m[kept]
    probs=np.load(CACHE/'clean_v6_causal_probs_full.npy')
    pol.THETAS=THETAS; pol.POLICIES=[(1,5)]
    val_best=None; val_rows=[]
    rr_v,yy_v,ss_v=rids[va_red],labels[va_red],symbols[va_red]
    for cool in COOLDOWNS:
        rows=evaluate_sequence_set('validation',rr_v,yy_v,probs[va_red],ss_v,DEFAULT_GT_GAP,cool)
        for r in rows:
            val_rows.append({'cooldown':cool,**r})
            if val_best is None or r['event_f1']>val_best['event_f1']:
                val_best={'cooldown':cool,**r}
    rr_t,yy_t,ss_t=rids[te_red],labels[te_red],symbols[te_red]
    test_rows=evaluate_sequence_set('test',rr_t,yy_t,probs[te_red],ss_t,DEFAULT_GT_GAP,val_best['cooldown'])
    test_sel=[r for r in test_rows if r['theta']==val_best['theta']][0]
    def row_compact(r):
        return {k:r.get(k) for k in ['theta','policy','gt_gap_beats','alert_cooldown_beats','global_auc','gt_events','alert_blocks','matched_gt_events','false_alarm_blocks','event_precision','event_recall','event_f1','fp_per_record','fp_per_1000_beats']}
    val_compact=row_compact(val_best); test_compact=row_compact(test_sel)
    # self-consistency
    assert val_compact['matched_gt_events'] <= val_compact['gt_events']
    assert val_compact['alert_blocks'] >= val_compact['false_alarm_blocks']
    if val_compact.get('matched_gt_events') is not None:
        assert val_compact['matched_gt_events'] >= 0
    assert test_compact['matched_gt_events'] <= test_compact['gt_events']
    assert test_compact['alert_blocks'] >= test_compact['false_alarm_blocks']
    json.dump({
        'date':time.strftime('%Y-%m-%d %H:%M:%S'),
        'model':str(MODELS_DIR/'ecg_model_exp7c_clean_v6_qat_int8.tflite'),
        'patient_stats':{k:pstats[k] for k in ['n_patients','n_train','n_val','n_test']},
        'validation_rows':[{'theta':x['theta'],'cooldown':x['cooldown'],'recall':x['event_recall'],'precision':x['event_precision'],'f1':x['event_f1'],'fp_per_record':x['fp_per_record']} for x in val_rows],
        'selected_on_validation':val_compact,
        'test_frozen':test_compact,
        'self_consistency':'PASS',
        'elapsed_s':round(time.time()-t0,1),
    },open(OUT,'w'),indent=2,ensure_ascii=False)
    print(json.dumps({'selected':val_compact,'test':test_compact},ensure_ascii=False,indent=2))
if __name__=='__main__': main()
