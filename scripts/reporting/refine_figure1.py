"""Ten documented design rounds for Figure 1, using verified metrics only."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from project_paths import REPORT_DIR, REPORT_SUPPORT, ARCHIVE_DIR

ROOT=REPORT_DIR
TRIALS=ARCHIVE_DIR/'figures/figure1_design_rounds'
NOTES={
1:'Move the original legend outside the axes to expose the She–They interval.',
2:'Align prompt and answer markers on one row per pair, simplifying pair matching.',
3:'Add faint connectors to guide the eye from each prompt to its answer distribution.',
4:'Order rows by observed mean answer distance: He–They, She–They, He–She.',
5:'Add exact four-decimal distances above markers, making close prompt values accessible.',
6:'Test separate prompt and response panels on an identical scale to separate quantities.',
7:'Test distinct, clearly labelled zoomed scales to inspect local differences in each panel.',
8:'Return to the connected single panel and test a zero-based axis for scale context.',
9:'Restore a single focused scale, and test report-width typography with the key above the axes.',
10:'Finalize a compact connected chart: labelled values, a shared scale, outside key, and extra space for intervals.',
}


def make(round_number):
    TRIALS.mkdir(parents=True, exist_ok=True)
    m=json.loads((REPORT_SUPPORT/'verified_metrics.json').read_text())
    keys=['he–she','he–they','she–they'] if round_number<4 else ['he–they','she–they','he–she']
    values=[m['pairs'][k] for k in keys]
    p=np.array([1-v['prompt_similarity'] for v in values])
    a=np.array([v['answer_distance'] for v in values])
    lo=np.array([1-v['similarity_ci'][1] for v in values])
    hi=np.array([1-v['similarity_ci'][0] for v in values])
    y=np.arange(3)
    colors=['#78828B','#205D83']
    fs=9 if round_number>=9 else 9.5
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':fs,'axes.spines.top':False,
                         'axes.spines.right':False,'savefig.facecolor':'white'})
    fig=plt.figure(figsize=(6.65,2.90))
    if round_number in [6,7]:
        axs=[fig.add_axes([.14,.28,.36,.48]),fig.add_axes([.60,.28,.36,.48])]
        for index,ax in enumerate(axs):
            if index==0:ax.scatter(p,y,s=40,marker='s',c=colors[0])
            else:ax.errorbar(a,y,xerr=[a-lo,hi-a],fmt='o',color=colors[1],capsize=3)
            ax.set(ylim=(2.5,-.5),yticks=y,yticklabels=[k.title() for k in keys] if index==0 else [],
                   xlim=(.175,.335),title=['Prompts','Answers: mean and 95% CI'][index])
            if round_number==7:
                ax.set_xlim((.187,.204) if index==0 else (.266,.331))
            ax.grid(axis='x',alpha=.15)
        fig.text(.5,.12,'Cosine distance (1 − similarity); smaller = closer',ha='center',fontsize=9)
        if round_number==7:fig.text(.5,.87,'Separate zoomed scales',ha='center',fontsize=10)
    else:
        bottom=.26 if round_number<9 else .24
        ax=fig.add_axes([.145,bottom,.825,.53 if round_number<9 else .52])
        if round_number==1:py,ay=y+.12,y-.12
        else:py,ay=y,y
        if round_number>=3:
            ax.hlines(y,p,a,color='#CED3D8',linewidth=1.4,zorder=1)
        ax.scatter(p,py,s=36,marker='s',color=colors[0],zorder=3)
        ax.errorbar(a,ay,xerr=[a-lo,hi-a],fmt='o',markersize=5.5,
                    color=colors[1],capsize=3,elinewidth=1.4,zorder=3)
        if round_number>=5:
            for pp,aa,yy in zip(p,a,y):
                ax.annotate(f'{pp:.4f}',(pp,yy),xytext=(0,10),textcoords='offset points',
                            ha='center',va='bottom',fontsize=8.5,color='#454B50')
                ax.annotate(f'{aa:.4f}',(aa,yy),xytext=(0,10),textcoords='offset points',
                            ha='center',va='bottom',fontsize=8.5,color=colors[1])
        ax.set(ylim=(2.55,-.65),yticks=y,yticklabels=[k.title() for k in keys],xlim=(.175,.335))
        if round_number==8:ax.set_xlim(0,.36)
        ax.set_xticks([.18,.20,.22,.24,.26,.28,.30,.32])
        if round_number==8:ax.set_xticks([0,.10,.20,.30])
        if round_number==10:ax.set_xticks([.18,.20,.22,.24,.26,.28,.30,.32])
        ax.grid(axis='x',color='#E5E7EB',linewidth=.6,zorder=0)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_color('#747B81')
        ax.tick_params(axis='y',length=0,pad=8)
        ax.tick_params(axis='x',length=3,color='#747B81')
        ax.set_xlabel('Cosine distance (1 − similarity)',fontsize=9,labelpad=7)
        handles=[Line2D([],[],marker='s',linestyle='',color=colors[0],markersize=5,label='Prompt'),
                 Line2D([],[],marker='o',linestyle='-',color=colors[1],markersize=5,label='Answer mean and 95% CI')]
        if round_number<9:
            fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.54,.01),ncol=2,frameon=False,fontsize=8)
        else:
            fig.legend(handles=handles,loc='upper right',bbox_to_anchor=(.98,.98),ncol=2,
                       frameon=False,fontsize=8.5,columnspacing=1.4,handletextpad=.5)
            fig.text(.145,.81,'Smaller distance = closer representations',fontsize=8.5,color='#454B50')
        if round_number==10:
            # Give direct labels breathing room at the top while retaining the
            # same 6.65-inch width and 2.90-inch height used in the document.
            ax.set_ylim(2.52,-.69)
            ax.tick_params(axis='y',labelsize=10)
            for label in ax.texts:
                label.set_fontsize(9)
            fig.text(.145,.925,'25 runs per pronoun',fontsize=8.5,color='#454B50')
    dest=TRIALS/f'round_{round_number:02d}.png'
    fig.savefig(dest,dpi=300)
    if round_number==10:
        fig.savefig(ROOT/'figures/distances.png',dpi=300)
        fig.savefig(ROOT/'figures/distances.pdf')
    plt.close(fig)
    (TRIALS/f'round_{round_number:02d}.txt').write_text(NOTES[round_number]+'\n')
    print(f'Round {round_number}: {NOTES[round_number]}')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('rounds',nargs='+',type=int,choices=range(1,11))
    for number in parser.parse_args().rounds:make(number)
