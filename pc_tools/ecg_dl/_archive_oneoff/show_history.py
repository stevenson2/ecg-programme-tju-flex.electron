import csv
rows = list(csv.DictReader(open('models/train_history.csv')))
for r in rows:
    ep = r['epoch']
    tl = float(r['loss'])
    ta = float(r['auc'])
    vl = float(r['val_loss'])
    va = float(r['val_auc'])
    print(f'ep{ep:>3} | tr_loss {tl:.4f} | tr_auc {ta:.4f} | val_loss {vl:.4f} | val_auc {va:.4f}')