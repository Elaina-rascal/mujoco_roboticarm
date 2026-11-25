"""用 ik-comparison.json 绘制静态竖版对比；数据来自离线 IK。"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch

P=Path(__file__).resolve().parent
D=json.loads((P/'ik-comparison.json').read_text())
a,b=D['results'];limits=np.array(D['joint_limits_deg'])
fp='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
font_manager.fontManager.addfont(fp)
plt.rcParams.update({'font.family':font_manager.FontProperties(fname=fp).get_name(),
 'font.size':11,'axes.unicode_minus':False,'svg.fonttype':'path','pdf.fonttype':42})
BEFORE='#C56950';AFTER='#228C7B';INK='#19364A';MUTED='#65788A';GRID='#E8EEF2'
fig=plt.figure(figsize=(9,14.2),facecolor='white')
def text(x,y,s,size=12,color=INK,weight='normal',**kwargs):
 fig.text(x,y,s,fontsize=size,color=color,weight=weight,**kwargs)
text(.075,.958,'放开 yaw，误差与关节构型如何改变',22,weight='bold')
text(.075,.928,'同一初值 · 同一目标位置 [20, 10, 45] cm · 六维固定姿态 vs 五维任务约束',10.5,MUTED)

def metric(x,title,v1,v2,unit):
 card=FancyBboxPatch((x,.824),.405,.079,transform=fig.transFigure,
  boxstyle='round,pad=0.012,rounding_size=0.012',fc='#F4F8FA',ec='none',zorder=0)
 fig.add_artist(card)
 text(x+.012,.878,title,12,MUTED)
 text(x+.012,.842,v1,24,BEFORE,weight='bold')
 text(x+.145,.846,'→',22,MUTED)
 text(x+.193,.842,v2,24,AFTER,weight='bold')
 text(x+.351,.847,unit,11,MUTED)
metric(.075,'末端位置距离','30.12','0.0485','cm')
metric(.525,'工具倾斜角','22.14','0.0439','°')
text(.075,.781,'01  完整 Log6 误差',16,weight='bold')
text(.075,.756,'● 六维固定姿态：迭代停滞',10.5,BEFORE)
text(.51,.756,'● 五维任务约束：收敛',10.5,AFTER)

values0=np.array(a['log6_display']);values1=np.array(b['log6_display'])
for column,(start,title,unit) in enumerate(((0,'平移部分 ρ','cm'),(3,'旋转部分 φ','°'))):
 ax=fig.add_axes([.115+column*.49,.578,.345,.139])
 v0,v1=values0[start:start+3],values1[start:start+3]
 bound=max(abs(v0).max(),abs(v1).max())*1.42
 ax.set_xlim(-bound,bound);ax.set_ylim(2.6,-.6)
 if start==3:ax.axhspan(1.55,2.45,color='#FFF5DE',zorder=0)
 y=np.arange(3)
 ax.barh(y-.15,v0,height=.24,color=BEFORE,zorder=3)
 ax.barh(y+.15,v1,height=.24,color=AFTER,zorder=3)
 ax.axvline(0,color='#A8B8C4',lw=.9)
 labels=[('ρ' if start==0 else 'φ')+'_'+c for c in 'xyz']
 ax.set_yticks(y,labels);ax.tick_params(axis='y',length=0,pad=7,labelsize=11)
 ax.tick_params(axis='x',length=0,labelsize=8,colors=MUTED)
 ax.set_title(title+' ('+unit+')',loc='left',fontsize=11,color=INK,pad=13)
 ax.xaxis.grid(True,color=GRID,zorder=0);ax.set_axisbelow(True)
 for spine in ax.spines.values():spine.set_visible(False)
 for vals,dy,col in ((v0,-.15,BEFORE),(v1,.15,AFTER)):
  for i,v in enumerate(vals):
   offset=bound*.04 if v>=0 else -bound*.04
   ax.text(v+offset,i+dy,f'{v:+.3f}',va='center',ha='left' if v>=0 else 'right',fontsize=8.4,color=col)
text(.075,.538,'黄色行：五维任务释放 yaw，不要求完整旋转误差归零。',10.5,MUTED)
text(.075,.519,'ρ 为 SE(3) 对数的平移部分；前三项用 cm，后三项用 °，不是欧拉角。',9.5,MUTED)

text(.075,.474,'02  六个关节的最终构型',16,weight='bold')
text(.075,.448,'灰色区间：关节允许范围',10,MUTED)
text(.405,.448,'● 六维末态',10,BEFORE)
text(.59,.448,'● 五维末态',10,AFTER)
ax=fig.add_axes([.145,.123,.565,.294])
ax.set_xlim(-185,185);ax.set_ylim(5.7,-.7)
q0=np.array(a['q_deg']);q1=np.array(b['q_deg']);y=np.arange(6)
for i,(lo,hi) in enumerate(limits):
 ax.plot([lo,hi],[i,i],color='#E5EBEF',lw=13,solid_capstyle='round',zorder=1)
 ax.plot([lo,lo],[i-.17,i+.17],color='#BAC7D0',lw=1.1)
 ax.plot([hi,hi],[i-.17,i+.17],color='#BAC7D0',lw=1.1)
 ax.plot([q0[i],q1[i]],[i,i],color='#A5B6C0',lw=1.4,zorder=2)
ax.scatter(q0,y-.075,s=73,color=BEFORE,edgecolor='white',linewidth=.9,zorder=4)
ax.scatter(q1,y+.075,s=73,color=AFTER,edgecolor='white',linewidth=.9,zorder=4)
for i in (0,4):
 ax.annotate('触限',xy=(q0[i],i-.075),xytext=(q0[i]-10,i-.36),
             fontsize=8,color=BEFORE,ha='right')
ax.set_yticks(y,[f'J{i}' for i in range(1,7)])
ax.tick_params(axis='y',length=0,pad=13,labelsize=12,colors=INK)
ax.set_xticks([-180,-120,-60,0,60,120,180]);ax.tick_params(axis='x',length=0,labelsize=8,colors=MUTED)
ax.set_xlabel('关节角 (°)',fontsize=10,color=MUTED,labelpad=11)
ax.xaxis.grid(True,color=GRID);ax.set_axisbelow(True)
for spine in ax.spines.values():spine.set_visible(False)
# Fixed numeric columns retain readability even when markers overlap.
ax.text(1.11,1.045,'六维 / °',transform=ax.transAxes,ha='center',color=BEFORE,fontsize=10)
ax.text(1.36,1.045,'五维 / °',transform=ax.transAxes,ha='center',color=AFTER,fontsize=10)
for i in range(6):
 ax.text(1.11,i,f'{q0[i]:+.2f}',transform=ax.get_yaxis_transform(),ha='center',va='center',color=BEFORE,fontsize=10)
 ax.text(1.36,i,f'{q1[i]:+.2f}',transform=ax.get_yaxis_transform(),ha='center',va='center',color=AFTER,fontsize=10)
text(.075,.066,'六维末态：J1、J5 触及限位；五维解：六个关节均在限位内。',11,INK)
text(.075,.039,'离线 IK 对照，非真机测量。五维只约束到点与水平，最终 yaw ≈ 26.40°。',9.5,MUTED)
fig.savefig(P/'ik-before-after-portrait.png',dpi=240,facecolor='white')
fig.savefig(P/'ik-before-after-portrait.svg',facecolor='white')
fig.savefig(P/'ik-before-after-portrait.pdf',facecolor='white')
print('Exported portrait PNG / SVG / PDF')
