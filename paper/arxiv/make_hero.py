import json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, 'results', 'exp1')
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(OUT, exist_ok=True)
def load(n):
    with open(os.path.join(RES, n)) as fh:
        return json.load(fh)
q = load('quality_power.json')['recall']
asym = load('ppl_asym.json')
tp = load('refresh_throughput.json')
PURPLE='#6c3fc5'; TEAL='#1f9e8e'; RED='#cc4444'; GREY='#b9b9c4'; ORANGE='#e8743b'
plt.rcParams.update({'font.size':9,'axes.edgecolor':'#888','axes.linewidth':0.8})
fig = plt.figure(figsize=(11, 4.3))
gs = fig.add_gridspec(1, 3, wspace=0.34, left=0.06, right=0.985, top=0.78, bottom=0.15)
def badge(ax, letter, title):
    ax.annotate(letter, xy=(0.0,1.17), xycoords='axes fraction', fontsize=12, fontweight='bold', color='white', ha='center', va='center', bbox=dict(boxstyle='round,pad=0.35', fc=PURPLE, ec='none'))
    ax.annotate(title, xy=(0.08,1.17), xycoords='axes fraction', fontsize=11, fontweight='bold', ha='left', va='center')
axA = fig.add_subplot(gs[0,0])
bits=[16,8,4,3]
acc=[q['bits%d'%b]['acc'] for b in bits]
lo=[acc[i]-q['bits%d'%b]['ci95_lo'] for i,b in enumerate(bits)]
hi=[q['bits%d'%b]['ci95_hi']-acc[i] for i,b in enumerate(bits)]
xs=np.arange(4)
axA.bar(xs,acc,color=[TEAL,TEAL,RED,RED],width=0.62,zorder=2)
axA.errorbar(xs,acc,yerr=[lo,hi],fmt='none',ecolor='black',capsize=4,lw=1,zorder=3)
for x,a in zip(xs,acc):
    axA.text(x,a+0.05,format(a,'.2f'),ha='center',fontsize=9,fontweight='bold')
axA.set_xticks(xs); axA.set_xticklabels(['16-bit','8-bit','4-bit','3-bit'])
axA.set_ylim(0,1.18); axA.set_ylabel('NIAH recall (n=60, Wilson CI)')
axA.set_xlabel('recurrent-state bit-width')
axA.grid(True,axis='y',alpha=0.3)
badge(axA,'A','The 8-bit Cliff')
axB = fig.add_subplot(gs[0,1]); axB.axis('off')
axB.set_xlim(0,10); axB.set_ylim(0,10)
def box(x,y,w,h,text,fc,ec):
    axB.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.12',fc=fc,ec=ec,lw=1.3))
    axB.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=8)
def arr(x1,y1,x2,y2,style='-'):
    axB.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=12,color='#555',lw=1.3,linestyle=style))
box(0.3,6.4,3.2,1.6,'full-precision\nrecurrent state','#efe7fb',PURPLE)
box(6.5,6.4,3.2,1.6,'quantize state\n(8 / 4 / 3-bit)','#e7f6f3',TEAL)
box(6.5,3.2,3.2,1.6,'generate next\ntoken','#ededf2','#888')
box(0.3,3.2,3.2,1.6,'every k steps:\nrefresh to FP','#fdeee6',ORANGE)
box(3.1,0.3,3.8,1.5,'error contained\n(no compounding)','#efe7fb',PURPLE)
arr(3.5,7.2,6.5,7.2)
arr(8.1,6.4,8.1,4.8)
arr(6.5,4.0,3.5,4.0)
arr(1.9,4.8,1.9,6.4,style='--')
arr(1.9,3.2,4.2,1.8)
arr(7.0,3.2,5.8,1.8)
badge(axB,'B','Quantize - Generate - Refresh')
axC = fig.add_subplot(gs[0,2])
pts = [('FP16 baseline', tp['baseline_int8_equiv']['tok_per_s'], asym['symmetric']['bits16'], PURPLE), ('8-bit (asym)', tp['k16']['tok_per_s'], asym['asymmetric']['bits8'], TEAL), ('4-bit (asym)', tp['k32']['tok_per_s'], asym['asymmetric']['bits4'], RED), ('3-bit (asym)', tp['k64']['tok_per_s'], asym['asymmetric']['bits3'], ORANGE)]
for name,x,y,c in pts:
    axC.scatter(x,y,s=90,color=c,zorder=3,edgecolor='white',lw=1)
    axC.annotate(name,(x,y),xytext=(6,6),textcoords='offset points',fontsize=7.5,color=c)
axC.set_yscale('log')
axC.set_xlabel('decode throughput (tokens/s)')
axC.set_ylabel('perplexity (log, lower=better)')
axC.grid(True,alpha=0.3)
badge(axC,'C','The Cost-Quality Pareto')
fig.suptitle('Quantizing the Mamba recurrent state: matching distributions, containing error', fontsize=12.5, fontweight='bold', x=0.06, ha='left', y=0.965)
fig.savefig(os.path.join(OUT,'hero_qstate.pdf'))
fig.savefig(os.path.join(OUT,'hero_qstate.png'),dpi=150)
print('WROTE', os.path.join(OUT,'hero_qstate.pdf'))
