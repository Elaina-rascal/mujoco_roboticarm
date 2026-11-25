"""简洁竖版：完整 Log6 误差在上，任务结论在下。"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch

P=Path(__file__).resolve().parent
D=json.loads((P/'ik-comparison.json').read_text());a,b=D['results']
font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
font_manager.fontManager.addfont(font)
plt.rcParams.update({'font.family':font_manager.FontProperties(fname=font).get_name(),
 'axes.unicode_minus':False,'svg.fonttype':'path','pdf.fonttype':42})
red='#C56950';green='#228C7B';ink='#19364A';muted='#6B7D8B'
fig=plt.figure(figsize=(8,8.4),facecolor='white')
fig.text(.09,.94,'放开 yaw 前后的 Log6 误差',fontsize=20,color=ink,weight='bold')
fig.text(.09,.892,'● 六维固定姿态',fontsize=12,color=red)
fig.text(.53,.892,'● 五维任务约束',fontsize=12,color=green)
x0=np.array(a['log6_display']);x1=np.array(b['log6_display'])
for i,(start,title,unit) in enumerate(((0,'平移分量','cm'),(3,'旋转分量','°'))):
 ax=fig.add_axes([.15,.575-i*.34,.74,.235])
 y=np.arange(3);v0=x0[start:start+3];v1=x1[start:start+3]
 bound=max(abs(v0).max(),abs(v1).max())*1.36
 ax.set_xlim(-bound,bound);ax.set_ylim(2.55,-.55)
 if start==3:ax.axhspan(1.55,2.45,color='#FFF3D5',zorder=0)
 ax.barh(y-.145,v0,height=.25,color=red,zorder=3)
 ax.barh(y+.145,v1,height=.25,color=green,zorder=3)
 ax.axvline(0,color='#A9B8C4',lw=.8)
 ax.set_yticks(y,[(('ρ' if start==0 else 'φ')+'_'+v) for v in 'xyz'])
 ax.tick_params(axis='y',length=0,labelsize=12,pad=10)
 ax.tick_params(axis='x',length=0,labelsize=9,colors=muted)
 ax.set_title(f'{title} ({unit})',loc='left',fontsize=12,color=ink,pad=14)
 ax.xaxis.grid(True,color='#E8EEF2');ax.set_axisbelow(True)
 for spine in ax.spines.values():spine.set_visible(False)
 for vals,delta,color in ((v0,-.145,red),(v1,.145,green)):
  for j,v in enumerate(vals):
   ax.text(v+(bound*.025 if v>=0 else -bound*.025),j+delta,f'{v:+.3f}',
    ha='left' if v>=0 else 'right',va='center',fontsize=10,color=color)
card=FancyBboxPatch((.075,.045),.85,.105,transform=fig.transFigure,
 boxstyle='round,pad=0.012,rounding_size=0.015',fc='#EFF7F4',ec='none')
fig.add_artist(card)
fig.text(.10,.113,'位置误差 30.12 → 0.0485 cm',fontsize=17,color=ink,weight='bold')
fig.text(.10,.073,'放开 yaw，位置与水平约束得以满足。',fontsize=13,color=green)
fig.text(.09,.019,'同一目标、同一初值的离线 IK；黄色分量不要求归零。',fontsize=9,color=muted)
for suffix in ('png','svg','pdf'):
 fig.savefig(P/f'ik-error-compact.{suffix}',dpi=240,facecolor='white')
print('Exported compact comparison')
