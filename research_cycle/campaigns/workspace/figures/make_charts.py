#!/usr/bin/env python3
import os, math
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({"font.size":12,"axes.titlesize":13,"axes.titleweight":"bold","figure.dpi":140,
                     "axes.grid":True,"grid.alpha":0.25,"axes.spines.top":False,"axes.spines.right":False})
OUT="/tmp/claude-1000/-home-friemann-workspace-maestratica/b9bd7464-4c05-40a5-ba2b-b36c0ae2836c/scratchpad/charts"
os.makedirs(OUT,exist_ok=True)
TEAL="#1b9e8a"; GRAY="#9aa0a6"; ORANGE="#e07b39"; BLUE="#3b6ea5"; RED="#c0392b"; GREEN="#2e8b57"

# 1 — ошибка видна внутри раньше, чем в ответе
fig,ax=plt.subplots(figsize=(6.6,4.0))
b=ax.bar(["по «нутру»\n(взгляд внутрь)","по «голосу»\n(ответ модели)"],[0.83,0.60],color=[TEAL,GRAY],width=0.55)
ax.axhline(0.5,ls="--",color="k",alpha=0.4); ax.text(1.35,0.51,"случайно",fontsize=10,alpha=0.6)
ax.set_ylim(0.4,0.95); ax.set_ylabel("точность ловли ошибки")
ax.set_title("Ошибка видна ВНУТРИ раньше, чем в ответе")
for r,v in zip(b,[0.83,0.60]): ax.text(r.get_x()+r.get_width()/2,v+0.01,f"{v:.2f}",ha="center",fontweight="bold")
fig.tight_layout(); fig.savefig(f"{OUT}/1_oshibka.png"); plt.close(fig)

# 2 — сжатие: слепая статистика бьёт «умный» разбор
k=[1,2,4,8,16,32,64]; smart=[.17,.17,.17,.31,.91,.95,.96]; blind=[.60,.93,.93,.96,.96,.96,.96]
fig,ax=plt.subplots(figsize=(6.8,4.0))
ax.plot(k,blind,"-o",color=BLUE,lw=2.4,label="вслепую (просто «громкие»)")
ax.plot(k,smart,"-o",color=ORANGE,lw=2.4,label="с умом (наша лупа)")
ax.set_xscale("log",base=2); ax.set_xticks(k); ax.set_xticklabels(k)
ax.set_xlabel("сколько направлений оставили при сжатии"); ax.set_ylabel("сохранился навык сложения")
ax.set_title("Сжатие: слепая статистика бьёт «умный» разбор"); ax.legend(loc="lower right")
fig.tight_layout(); fig.savefig(f"{OUT}/2_szhatie.png"); plt.close(fig)

# 3 — что решает, выживет ли связь при сжатии (как забывает)
labels=["популярность\nродителя","близость\nк стволу","размер\nветки"]; vals=[0.84,0.73,0.57]
fig,ax=plt.subplots(figsize=(6.6,4.0))
b=ax.bar(labels,vals,color=[GREEN,"#5aa469","#9ccca8"],width=0.6)
ax.axhline(0.5,ls="--",color="k",alpha=0.4); ax.text(2.0,0.51,"случайно",fontsize=10,alpha=0.6)
ax.set_ylim(0.4,0.95); ax.set_ylabel("сила предсказания «выживет ли связь»")
ax.set_title("Что мозг держит при сжатии: ствол и хабы, не веточки")
for r,v in zip(b,vals): ax.text(r.get_x()+r.get_width()/2,v+0.01,f"{v:.2f}",ha="center",fontweight="bold")
fig.tight_layout(); fig.savefig(f"{OUT}/3_zabyvanie.png"); plt.close(fig)

# 4 — сколько параметров нужно на дерево (замер vs теория)
V=[32,64,128,256]; meas=[308,460,1706,3370]; theory=[sum(math.log2(i) for i in range(1,v))/2 for v in V]
fig,ax=plt.subplots(figsize=(6.8,4.0))
ax.plot(V,meas,"-o",color=BLUE,lw=2.4,label="наш замер (крохотные сети)")
ax.plot(V,theory,"--o",color=GRAY,lw=2,label="теор. минимум (2 бита/парам)")
ax.set_xlabel("число видов в дереве"); ax.set_ylabel("нужно параметров")
ax.set_title("Сколько нейронов, чтобы хранить дерево ТОЧНО"); ax.legend(loc="upper left")
fig.tight_layout(); fig.savefig(f"{OUT}/4_emkost.png"); plt.close(fig)

# 5 — правило или зубрёжка: достроит ли невиданное
Vg=["64","128","256","512"]; reg=[1.00,0.75,0.66,0.69]; rnd=[0.00,0.03,0.02,0.00]
x=np.arange(len(Vg)); w=0.38
fig,ax=plt.subplots(figsize=(6.8,4.0))
ax.bar(x-w/2,reg,w,color=TEAL,label="узорчатое (есть правило)")
ax.bar(x+w/2,rnd,w,color=GRAY,label="случайное (нет правила)")
ax.set_xticks(x); ax.set_xticklabels(Vg); ax.set_xlabel("число видов")
ax.set_ylabel("верно достроено из невиданного"); ax.set_ylim(0,1.05)
ax.set_title("Архивация = найти ПРАВИЛО (тогда достраивает невиданное)"); ax.legend(loc="upper right")
fig.tight_layout(); fig.savefig(f"{OUT}/5_pravilo.png"); plt.close(fig)

# 6 — иллюстрация: ствол держит, веточки теряет
fig,ax=plt.subplots(figsize=(6.8,4.2)); ax.axis("off")
# позиции узлов уровнями
levels={0:[(0.5,1.0)],1:[(0.28,0.72),(0.72,0.72)],2:[(0.16,0.44),(0.40,0.44),(0.60,0.44),(0.84,0.44)],
        3:[(0.10,0.16),(0.22,0.16),(0.34,0.16),(0.46,0.16),(0.54,0.16),(0.66,0.16),(0.78,0.16),(0.90,0.16)]}
def edge(a,b,kept):
    ax.plot([a[0],b[0]],[a[1],b[1]],("-" if kept else "--"),
            color=(GREEN if kept else RED),lw=(3.2 if kept else 1.6),alpha=(0.95 if kept else 0.8),zorder=1)
# ствол (уровни 0-1-2) держит, веточки (2-3) теряет
for i,ch in enumerate(levels[1]): edge(levels[0][0],ch,True)
for i,ch in enumerate(levels[2]): edge(levels[1][i//2],ch,True)
for i,ch in enumerate(levels[3]): edge(levels[2][i//2],ch,False)
for lv,pts in levels.items():
    for (x0,y0) in pts:
        kept = lv<=2
        ax.scatter([x0],[y0],s=(230 if lv<=1 else 150 if lv==2 else 80),
                   color=(GREEN if kept else RED),zorder=3,edgecolors="white",linewidths=1.2)
ax.plot([],[],"-",color=GREEN,lw=3,label="ствол — держит"); ax.plot([],[],"--",color=RED,lw=1.6,label="веточки — теряет")
ax.legend(loc="lower center",ncol=2,frameon=False,bbox_to_anchor=(0.5,-0.02))
ax.set_title("Малый мозг = сжатая картинка дерева:\nствол держит, мелкие веточки теряет и путает",pad=8)
ax.set_xlim(0,1); ax.set_ylim(0.02,1.1)
fig.tight_layout(); fig.savefig(f"{OUT}/6_derevo.png"); plt.close(fig)

print("готово:", sorted(os.listdir(OUT)))
