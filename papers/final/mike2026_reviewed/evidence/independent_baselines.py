"""Independent-input reference baselines; does not train or validate X-Fusion.

Uses only 16x16 grayscale pixels and recorded age/sex as predictors. Labels are
used exclusively as training/evaluation targets. Hyperparameters are selected
on validation patients, never on test data. No contaminated checkpoint is used.
"""
from pathlib import Path
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[k]='2'
import argparse, json, hashlib, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score,f1_score
from threadpoolctl import threadpool_limits

LABELS=['Atelectasis','Cardiomegaly','Effusion','Infiltration','Mass','Nodule','Pneumonia','Pneumothorax','Consolidation','Edema','Emphysema','Fibrosis','Pleural_Thickening','Hernia']

def run(data, dest):
    dest.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(data/'Data_Entry_2017_v2020.csv')
    available={p.name for p in (data/'images').glob('*.png')}
    df=df[df['Image Index'].isin(available)].copy().reset_index(drop=True)
    trainval=set((data/'train_val_list.txt').read_text().splitlines()); testlist=set((data/'test_list.txt').read_text().splitlines())
    patients=df.loc[df['Image Index'].isin(trainval),'Patient ID'].unique().copy()
    np.random.RandomState(42).shuffle(patients); vp=set(patients[:int(len(patients)*.1)])
    tr=df['Image Index'].isin(trainval)&~df['Patient ID'].isin(vp)
    va=df['Image Index'].isin(trainval)&df['Patient ID'].isin(vp)
    te=df['Image Index'].isin(testlist)
    masks={'train':tr.to_numpy(),'validation':va.to_numpy(),'test':te.to_numpy()}
    sets={k:set(df.loc[m,'Patient ID']) for k,m in masks.items()}
    assert not(sets['train']&sets['validation'] or sets['train']&sets['test'] or sets['validation']&sets['test'])
    assert np.all(sum(m.astype(int) for m in masks.values())==1)
    y=np.array([[int(k in str(v).split('|')) for k in LABELS] for v in df['Finding Labels']],dtype=np.int8)
    # Metadata transforms have no access to labels; fit imputation on train only.
    age=pd.to_numeric(df['Patient Age'],errors='coerce').to_numpy(float)
    age[(age<0)|(age>120)]=np.nan
    median=float(np.nanmedian(age[masks['train']]))
    age=np.nan_to_num(age,nan=median)
    sex=df['Patient Gender'].map({'M':1.,'F':0.}).fillna(.5).to_numpy()
    demo=np.column_stack([age,sex]).astype(np.float32)
    feature_file=dest/'image_features_16x16.npy'
    manifest_hash=hashlib.sha256('\n'.join(df['Image Index']).encode()).hexdigest()
    if feature_file.exists() and (dest/'feature_manifest_sha256.txt').read_text()==manifest_hash:
        pixels=np.load(feature_file)
    else:
        def image_feature(name):
            with Image.open(data/'images'/name) as im:
                return np.asarray(im.convert('L').resize((16,16),Image.Resampling.BILINEAR),dtype=np.float32).reshape(-1)/255.
        start=time.time(); features=[]
        with ThreadPoolExecutor(max_workers=8) as pool:
            for i,feature in enumerate(pool.map(image_feature,df['Image Index']),1):
                features.append(feature)
                if i%2000==0: print(f'Pixel extraction {i}/{len(df)} ({time.time()-start:.0f}s)',flush=True)
        pixels=np.stack(features);np.save(feature_file,pixels)
        (dest/'feature_manifest_sha256.txt').write_text(manifest_hash)
    report={'protocol':'Independent-input reference baselines; not X-Fusion validation','image_size':[16,16],'labels':LABELS,'seed':42,'split':{k:{'images':int(m.sum()),'patients':len(sets[k])} for k,m in masks.items()},'age_imputation_training_median':median,'manifest_sha256':manifest_hash,'models':{}}
    def score(target,p):
        f1_average='binary' if target.ndim==2 and target.shape[1]==1 else 'macro'
        return {'macro_auroc':float(roc_auc_score(target,p,average='macro')),'macro_ap':float(average_precision_score(target,p,average='macro')),'macro_f1_at_0_5':float(f1_score(target,p>=.5,average=f1_average,zero_division=0))}
    prior=y[masks['train']].mean(0)
    p=np.tile(prior,(masks['test'].sum(),1))
    report['models']['training_prevalence']={'test':score(y[masks['test']],p)}
    for title,x in [('age_sex',demo),('image_pixels',pixels),('image_pixels_age_sex',np.column_stack([pixels,demo]))]:
        scale=StandardScaler().fit(x[masks['train']]); x=scale.transform(x)
        choices=[]
        for c in [.01,.1,1.]:
            models=[]; converged=[]
            for k in range(len(LABELS)):
                model=LogisticRegression(C=c,solver='lbfgs',max_iter=500,random_state=42)
                model.fit(x[masks['train']],y[masks['train'],k]);models.append(model)
                converged.append(int(model.n_iter_[0])<500)
            val=np.column_stack([m.predict_proba(x[masks['validation']])[:,1] for m in models])
            metrics=score(y[masks['validation']],val)
            print(title,'C',c,'validation',metrics,flush=True)
            choices.append((metrics['macro_auroc'],c,models,metrics,all(converged)))
        _,c,models,validation,converged=max(choices,key=lambda v:v[0])
        pred=np.column_stack([m.predict_proba(x[masks['test']])[:,1] for m in models])
        result={'selected_C':c,'validation':validation,'test':score(y[masks['test']],pred),'all_selected_models_converged':converged,'per_label':{label:score(y[masks['test'],k:k+1],pred[:,k:k+1]) for k,label in enumerate(LABELS)}}
        report['models'][title]=result
        record=df.loc[masks['test'],['Image Index','Patient ID']].reset_index(drop=True)
        for k,label in enumerate(LABELS):record['label_'+label]=y[masks['test'],k];record['prob_'+label]=pred[:,k]
        record.to_csv(dest/(title+'_test_predictions.csv'),index=False)
        (dest/'results.json').write_text(json.dumps(report,indent=2))
        print(title,'TEST',result['test'],flush=True)
    df[['Image Index','Patient ID']].assign(split=np.select([masks['train'],masks['validation'],masks['test']],['train','validation','test'],default='unassigned')).to_csv(dest/'split_manifest.csv',index=False)
    (dest/'results.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    with threadpool_limits(limits=2):run(args.data,args.output)
