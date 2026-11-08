"""Reproduce the correct-conjugation report metrics and scientific figures."""
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer

import analyze_tfidf_biases as tf
from project_paths import PRIMARY_DATA, REPORT_DIR, REPORT_SUPPORT

OUT = REPORT_DIR
COLORS = {'he': '#AA4672', 'she': '#2876A8', 'they': '#248677'}
SUBJECTS = ['he', 'she', 'they']


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT_SUPPORT.mkdir(parents=True, exist_ok=True)
    (OUT / 'figures').mkdir(exist_ok=True)
    records = tf.read_records(PRIMARY_DATA)
    by_subject = defaultdict(dict)
    prompts = {}
    for run, number, prompt, answer in records:
        s = tf.extract_subject(prompt, number)
        assert s in SUBJECTS and run not in by_subject[s]
        by_subject[s][run] = answer
        prompts[s] = prompt
    runs = sorted(by_subject['he'])
    assert len(runs) == 25 and all(sorted(by_subject[s]) == runs for s in SUBJECTS)
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2', local_files_only=True)
    pe = model.encode([prompts[s] for s in SUBJECTS], normalize_embeddings=True)
    emb = {s: model.encode([by_subject[s][r] for r in runs], normalize_embeddings=True) for s in SUBJECTS}
    means = {s: emb[s].mean(axis=0) / np.linalg.norm(emb[s].mean(axis=0)) for s in SUBJECTS}
    rng = np.random.default_rng(20261108)
    bootstrap_indices = rng.integers(0, 25, (20000, 25))
    pairs = [('he', 'she'), ('he', 'they'), ('she', 'they')]
    metrics = {'n_runs': 25, 'prompts': prompts, 'pairs': {}, 'lexical': {}, 'bootstrap_seed': 20261108}
    scores = {}
    for a,b in pairs:
        v = np.sum(emb[a] * emb[b], axis=1)
        scores[(a,b)] = v
        ci = np.quantile(v[bootstrap_indices].mean(axis=1), [.025,.975])
        metrics['pairs'][a+'–'+b] = dict(prompt_similarity=float(pe[SUBJECTS.index(a)] @ pe[SUBJECTS.index(b)]),
            answer_similarity=float(v.mean()), answer_sd=float(v.std(ddof=1)),
            answer_distance=float(1-v.mean()), centroid_similarity=float(means[a] @ means[b]),
            similarity_ci=ci.tolist())
    contrast = scores[('he','they')] - scores[('she','they')]
    metrics['he_they_minus_she_they'] = {'mean':float(contrast.mean()),
        'bootstrap_ci':np.quantile(contrast[bootstrap_indices].mean(axis=1), [.025,.975]).tolist()}
    rows = tf.analyze(records, 25, True, False, 100000)
    with (REPORT_SUPPORT/'tfidf_all_terms.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    surface_counts = {}
    for s in SUBJECTS:
        texts = list(by_subject[s].values())
        tokens = [tf.normalized_tokens(x, True, False)[0] for x in texts]
        counts=Counter(w for tok in tokens for w in tok)
        surface_counts[s]=Counter(w for x in texts for w in tf.TOKEN_PATTERN.findall(x.lower()))
        metrics['lexical'][s] = {'retained_tokens':sum(counts.values()),'unique_answers':len(set(texts)),
            'selected_stem_counts':{tf.STEMMER.stem(w):counts[tf.STEMMER.stem(w)] for w in
                ['insecurity','drama','entertaining','capture','attention','facts','truth','command','likely']},
            'insecurity_response_count':sum(tf.STEMMER.stem('insecurity') in tok for tok in tokens),
            'truth_surface_count':surface_counts[s]['truth']}
    # Sensitivity check: build n-grams within responses rather than across them.
    safe = {s:Counter() for s in SUBJECTS}
    for s in SUBJECTS:
        for x in by_subject[s].values():
            tokens,_=tf.normalized_tokens(x,True,False)
            safe[s].update(tf.make_ngrams(tokens,2))
    metrics['within_response_bigrams']={s:dict(safe[s].most_common(15)) for s in SUBJECTS}
    safe_rows=[]
    for s in SUBJECTS:
        for term,count in safe[s].items():
            df=sum(term in safe[other] for other in SUBJECTS)
            safe_rows.append({'subject':s,'ngram':term,'count':count,
                'tf_idf':float(count/sum(safe[s].values())*np.log(3/df))})
    safe_rows.sort(key=lambda r:(r['subject'],-r['tf_idf'],r['ngram']))
    metrics['within_response_top_bigrams']={s:[r for r in safe_rows if r['subject']==s][:10] for s in SUBJECTS}
    (REPORT_SUPPORT/'verified_metrics.json').write_text(json.dumps(metrics,indent=2))
    np.savez(REPORT_SUPPORT/'response_embeddings.npz', **emb, prompt_embeddings=pe)

    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    from .refine_figure1 import make as make_figure1
    make_figure1(10)
    save_lexical_figures(rows, safe_rows, metrics)
    print(json.dumps(metrics,indent=2))


def save_lexical_figures(rows, safe_rows, metrics):
    """Render the report's lexical figures independently of model inference."""
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
        'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
    # Ranked bars preserve exact scores; labels show readable surface forms.
    aliases={'like':'likely','imag':'image','insecur':'insecurity family','sens':'sense','captiv':'captivating',
             'entertain':'entertaining','emphas':'emphasize','memor':'memorable','intens':'intensity',
             'otherwis':'otherwise','storytel':'storytelling','under':'underlying', 'fact':'facts',
             'simpli':'simply','eye':'eyes','attent':'attention','import':'importance','stori':'story',
             'exagger':'exaggerate','captur':'capture','signific':'significance','emot':'emotion',
             'other':'others'}
    for kind in ['unigram','bigram']:
        fig,axes=plt.subplots(1,3,figsize=(11.5,4.5),layout='constrained',sharex=True)
        for ax,s in zip(axes,SUBJECTS):
            top=([r for r in rows if r['subject']==s and r['ngram_type']==kind][:8]
                 if kind=='unigram' else [r for r in safe_rows if r['subject']==s][:8])
            labels=[aliases.get(r['ngram'],r['ngram']) for r in top]
            if kind=='bigram':
                labels=[' → '.join(aliases.get(word,word) for word in r['ngram'].split()) for r in top]
            ax.barh(np.arange(len(top)),[r['tf_idf'] for r in top],color=COLORS[s])
            ax.set(yticks=np.arange(len(top)),yticklabels=labels,title=s.capitalize(),xlabel='TF-IDF')
            ax.invert_yaxis();ax.grid(axis='x',alpha=.15)
        fig.savefig(OUT/f'figures/{kind}_rankings.png',dpi=240);plt.close(fig)
    selected=['insecurity','drama','entertaining','facts','truth','command']
    fig,ax=plt.subplots(figsize=(8.5,3.8),layout='constrained')
    for i,s in enumerate(SUBJECTS):
        counts=[metrics['lexical'][s]['selected_stem_counts'][tf.STEMMER.stem(w)] for w in selected]
        bars=ax.bar(np.arange(6)+(i-1)*.24,counts,width=.24,color=COLORS[s],label=s.capitalize())
        ax.bar_label(bars,fontsize=8,padding=2)
    ax.set(xticks=np.arange(6),xticklabels=['Insecurity\nfamily','Drama','Entertaining','Facts','Truth','Command'],
           ylabel='Occurrences across 25 answers',ylim=(0,16))
    ax.legend(frameon=False,ncol=3);ax.grid(axis='y',alpha=.15)
    fig.savefig(OUT/'figures/lexical_counts.png',dpi=240);plt.close(fig)


if __name__=='__main__': main()
