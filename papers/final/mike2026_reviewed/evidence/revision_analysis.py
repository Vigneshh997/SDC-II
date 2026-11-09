"""Recompute disease metrics and patient-cluster bootstrap intervals from saved predictions."""
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out=Path(__file__).resolve().parent.parent;out.mkdir(exist_ok=True)
evidence=out/'evidence';evidence.mkdir(exist_ok=True)
src=evidence
old=json.loads((src/'results.json').read_text());labels=old['labels']
models=['image_pixels','image_pixels_age_sex']
frames={m:pd.read_csv(src/(m+'_test_predictions.csv')) for m in models}
base=frames[models[0]]
assert all(base['Image Index'].equals(df['Image Index']) and base['Patient ID'].equals(df['Patient ID']) for df in frames.values())
y=np.column_stack([base['label_'+k].to_numpy(int) for k in labels])
patients,ids=np.unique(base['Patient ID'],return_inverse=True)
npatients=len(patients);rng=np.random.RandomState(20261015);B=500
def prepare(y,p):
    order=np.argsort(-p,kind='stable'); scores=p[order]
    ends=np.r_[np.flatnonzero(scores[1:]!=scores[:-1]),len(scores)-1]
    return order,ends,y[order]
def weighted_metric(cache,w):
    order,ends,ys=cache;ww=w[order]
    tp=np.cumsum(ww*ys)[ends];fp=np.cumsum(ww*(1-ys))[ends]
    pos=tp[-1];neg=fp[-1]
    if pos==0 or neg==0:return np.nan,np.nan
    prev_tp=np.r_[0,tp[:-1]];prev_fp=np.r_[0,fp[:-1]]
    auc=np.sum((fp-prev_fp)*(tp+prev_tp)/2)/(pos*neg)
    ap=np.sum((tp-prev_tp)*np.divide(tp,tp+fp,out=np.zeros_like(tp),where=(tp+fp)>0))/pos
    return float(auc),float(ap)
caches={};point={};draws={m:np.full((B,len(labels),2),np.nan) for m in models}
for m,df in frames.items():
    p=np.column_stack([df['prob_'+k] for k in labels]);caches[m]=[prepare(y[:,k],p[:,k]) for k in range(len(labels))]
    point[m]=np.array([weighted_metric(c,np.ones(len(y))) for c in caches[m]])
    for k in range(len(labels)):
        assert abs(point[m][k,0]-roc_auc_score(y[:,k],p[:,k]))<1e-12
        assert abs(point[m][k,1]-average_precision_score(y[:,k],p[:,k]))<1e-12
for b in range(B):
    counts=np.bincount(rng.randint(0,npatients,npatients),minlength=npatients)
    weights=counts[ids].astype(float)
    for m in models:
        draws[m][b]=[weighted_metric(c,weights) for c in caches[m]]
records=[]
for k,label in enumerate(labels):
    for m in models:
        finite=np.isfinite(draws[m][:,k,0]);valid=draws[m][finite,k,:]
        lower,upper=np.percentile(valid,[2.5,97.5],axis=0)
        records.append({'model':m,'finding':label,'positive_images':int(y[:,k].sum()),'positive_patients':int(len(np.unique(base.loc[y[:,k]==1,'Patient ID']))),'auroc':point[m][k,0],'auroc_lower95':lower[0],'auroc_upper95':upper[0],'average_precision':point[m][k,1],'ap_lower95':lower[1],'ap_upper95':upper[1],'valid_bootstraps':int(finite.sum())})
table=pd.DataFrame(records);table.to_csv(evidence/'disease_metrics_cluster_bootstrap.csv',index=False)
valid=np.all(np.isfinite(draws[models[0]][:,:,0])&np.isfinite(draws[models[1]][:,:,0]),axis=1)
delta=draws[models[1]][valid,:,0].mean(1)-draws[models[0]][valid,:,0].mean(1)
summary={'method':'Percentile patient-cluster bootstrap; sample patients with replacement and retain all their image rows by multiplicity','resamples':B,'seed':20261015,'test_images':len(y),'test_patients':npatients,'macro_auroc_delta_combined_minus_pixels':float(point[models[1]][:,0].mean()-point[models[0]][:,0].mean()),'macro_auroc_delta_lower95':float(np.percentile(delta,2.5)),'macro_auroc_delta_upper95':float(np.percentile(delta,97.5)),'valid_macro_delta_resamples':int(valid.sum()),'input_sha256':{m:hashlib.sha256((src/(m+'_test_predictions.csv')).read_bytes()).hexdigest() for m in models},'pneumonia_counts':table[table.finding=='Pneumonia'][['positive_images','positive_patients']].iloc[0].to_dict()}
(evidence/'uncertainty_summary.json').write_text(json.dumps(summary,indent=2))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10.5,'pdf.fonttype':42,'ps.fonttype':42})
fig,axes=plt.subplots(1,2,figsize=(6.8,4.9),sharey=True,gridspec_kw={'width_ratios':[1,1]})
pos=np.arange(len(labels));colors=['#2668a6','#bd5726'];names=['Image pixels','Pixels + age/sex']
for ax,metric,lo,hi,title in [(axes[0],'auroc','auroc_lower95','auroc_upper95','AUROC'),(axes[1],'average_precision','ap_lower95','ap_upper95','Average precision')]:
    for j,m in enumerate(models):
        vals=table[table.model==m].set_index('finding').loc[labels]
        xx=vals[metric].to_numpy();low=vals[lo].to_numpy();high=vals[hi].to_numpy()
        # Bootstrap intervals need not contain the original point estimate.
        yy=pos+(j-.5)*.20
        ax.hlines(yy,low,high,color=colors[j],linewidth=1,alpha=.8)
        ax.scatter(xx,yy,s=19,color=colors[j],marker=['o','s'][j],label=names[j],zorder=3)
    ax.set_xlim(0,1);ax.set_xticks([0,.5,1]);ax.grid(axis='x',alpha=.2);ax.set_title(title,fontweight='bold');ax.spines[['top','right']].set_visible(False)
axes[0].axvline(.5,color='#666666',linestyle='--',linewidth=.8)
axes[0].set_yticks(pos,[k.replace('_',' ')+f' ({int(y[:,i].sum())})' for i,k in enumerate(labels)])
axes[0].invert_yaxis();axes[0].set_ylabel('Finding (positive test images)')
fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.62,.00))
fig.subplots_adjust(left=.35,right=.98,top=.94,bottom=.13,wspace=.16)
fig.savefig(out/'per_disease_performance.pdf',bbox_inches='tight',metadata={'Author':'','Title':'Independent-input per-disease baseline performance'})
fig.savefig(out/'per_disease_performance.png',dpi=220,bbox_inches='tight')
print(json.dumps(summary,indent=2))
