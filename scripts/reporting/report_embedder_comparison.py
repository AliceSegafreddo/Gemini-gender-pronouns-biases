"""Compare prompt geometry across the four embedding models used in the study."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer
from project_paths import REPORT_DIR, REPORT_SUPPORT

MODELS=[
    ('sentence-transformers/all-MiniLM-L6-v2','MiniLM-L6'),
    ('sentence-transformers/all-MiniLM-L12-v2','MiniLM-L12'),
    ('sentence-transformers/paraphrase-MiniLM-L3-v2','paraphrase-MiniLM-L3'),
    ('BAAI/bge-small-en-v1.5','BGE-small-en'),
]
ROOT=REPORT_DIR


def main():
    original=json.loads((REPORT_SUPPORT/'verified_metrics.json').read_text())
    prompts=[original['prompts'][s] for s in ['he','she','they']]
    rows=[]
    for model_id,label in MODELS:
        model=SentenceTransformer(model_id,local_files_only=True,device='cpu')
        embeddings=model.encode(prompts,normalize_embeddings=True)
        d=lambda i,j:float(1-np.dot(embeddings[i],embeddings[j]))
        he_she,he_they,she_they=d(0,1),d(0,2),d(1,2)
        reference_mean=(he_they+she_they)/2
        rows.append(dict(model=model_id,label=label,he_she=he_she,he_they=he_they,
            she_they=she_they,absolute_gap=abs(she_they-he_they),
            signed_relative_gap_pct=100*(she_they-he_they)/reference_mean,
            absolute_relative_gap_pct=100*abs(she_they-he_they)/reference_mean))
        print(json.dumps(rows[-1]),flush=True)
    (REPORT_SUPPORT/'embedder_comparison.json').write_text(json.dumps({'prompts':prompts,'models':rows},indent=2))
    with (REPORT_SUPPORT/'embedder_comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    plot(rows)


def plot(rows):
    # Raw distances supply context; a relative contrast compares local balance
    # without treating distances from different models as calibrated units.
    rows=sorted(rows,key=lambda r:r['absolute_relative_gap_pct'])
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
                         'axes.spines.right':False})
    fig=plt.figure(figsize=(6.65,3.50))
    ax=fig.add_axes([.29,.24,.64,.57])
    y=np.arange(len(rows))
    for i,r in enumerate(rows):
        color='#205D83' if r['label']=='MiniLM-L12' else '#858C93'
        ax.plot([0,r['absolute_relative_gap_pct']],[i,i],color=color,lw=1.5)
        ax.scatter(r['absolute_relative_gap_pct'],i,s=40,color=color,zorder=3)
        x=r['absolute_relative_gap_pct']
        ax.annotate(f'{x:.2f}%',(x,i),xytext=(7,0),
            textcoords='offset points',ha='left',va='center',fontsize=8.5)
    extent=max(r['absolute_relative_gap_pct'] for r in rows)
    ax.set(xlim=(-1,extent*1.28),ylim=(len(rows)-.55,-.65),
        yticks=y,yticklabels=[r['label'] for r in rows],xlabel='Relative distance asymmetry (%)')
    ax.axvline(0,color='#353A40',lw=.8,zorder=0)
    ax.grid(axis='x',alpha=.15)
    ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0,pad=8)
    fig.text(.29,.92,'Prompt balance across embedding models',fontweight='bold',fontsize=10)
    fig.text(.29,.85,'Smaller absolute difference = greater balance',fontsize=8.5,color='#454B50')
    fig.text(.29,.10,'0% = equal distances to He and She',fontsize=8,color='#454B50')
    fig.text(.29,.035,'All four models place They closer to He.',fontsize=8,color='#454B50')
    fig.savefig(ROOT/'figures/embedder_comparison.png',dpi=300)
    fig.savefig(ROOT/'figures/embedder_comparison.pdf')
    plt.close(fig)


if __name__=='__main__':main()
