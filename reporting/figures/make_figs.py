import json, os, numpy as np, matplotlib
BASE=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","..")
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.spines.top":False,"axes.spines.right":False,"axes.edgecolor":"#8a8f98","axes.linewidth":0.8})
BL,OR,GR,INK="#2a78d6","#eb6834","#6b7280","#1f2328"
A=json.load(open(os.path.join(BASE,"a1_simulation","results","a1_summary_20261031_220543.json")))
st={e["bootstrap_stream"]:e for e in A["S_Q1_family"]}
def save(fig,name):
    fig.savefig(name+".png",dpi=300); fig.savefig(name+".tiff",dpi=600,pil_kwargs={"compression":"tiff_lzw"}); fig.savefig(name+".eps")
# ---------- Figure 1: design schematic (no data)
T=100; t=np.linspace(0,T,1001); q=np.clip((t-15)/60,0,1); th=1.2+(0.05-1.2)*q
fig=plt.figure(figsize=(6.8,4.6)); gs=fig.add_gridspec(4,1,height_ratios=[1.4,0.55,0.55,0.55],hspace=0.95)
ax=fig.add_subplot(gs[0]); ax.plot(t,th,color=INK,lw=1.6); ax.set_ylabel("θ(t)"); ax.set_xlim(0,T); ax.set_ylim(0,1.35)
ax.axvspan(15,75,color="#f1f3f6",zorder=0); ax.text(45,1.22,"slowing: θ falls from 1.2 to 0.05",ha="center",fontsize=8.5,color="#444")
ax.set_title("(a) Primary data-generating mechanism: Ornstein–Uhlenbeck process with slowing",loc="left",fontsize=9)
rng=np.random.default_rng(3); n=40; base=15*np.arange(1,9)/9; esc=55.0
# illustrative uneven times: rate 0.6 before esc, 1.2 after, thinned to n
pre=np.cumsum(rng.exponential(1/0.6,200)); pre=pre[(pre>15)&(pre<esc)]
post=esc+np.cumsum(rng.exponential(1/1.2,200)); post=post[post<T]
allinf=np.sort(np.r_[pre,post]); keep=np.sort(rng.choice(allinf,n-8,replace=False)); unev=np.r_[base,keep]
even=np.r_[base,np.linspace(15,T,n-8,endpoint=False)]; rnd=np.r_[base,np.sort(rng.uniform(15,T,n-8))]
for i,(name,tt,col,lab) in enumerate([("even",even,GR,"(b) Even: same count, evenly spaced"),("random",rnd,OR,"(c) Random-time: same count, irregular, not escalating"),("uneven",unev,BL,"(d) Uneven: same count, escalating after the trigger")]):
    a=fig.add_subplot(gs[i+1]); a.vlines(tt,0,1,color=col,lw=1.1); a.set_xlim(0,T); a.set_ylim(0,1); a.set_yticks([])
    a.set_title(lab,loc="left",fontsize=9)
    if name=="uneven": a.axvline(esc,color=INK,ls="--",lw=0.9); a.text(esc+1,0.55,"trigger (illustrative)",fontsize=7.5,color="#444")
    if i<2: a.set_xticklabels([])
a.set_xlabel("Time")
fig.subplots_adjust(left=0.08,right=0.98,top=0.95,bottom=0.09); save(fig,"Figure_1"); plt.close(fig)
# ---------- Figure 2: decomposition (F1) and escalation dose-response
fig,(a1,a2)=plt.subplots(1,2,figsize=(7.4,3.7),gridspec_kw={"width_ratios":[1.35,1]})
rows=[];labels=[]
for test,tl in (("ou_ss","Timing-aware"),("ar1","Timestamp-blind AR(1)")):
    for comp,cl in (("total(inf-reg)","Total"),("irregularity(pm-reg)","Irregularity"),("escalation(inf-pm)","Escalation")):
        e=st[f"bootstrap|ou_csd|A1|F1|Q1|{comp}|{test}|var"]; rows.append((e["diff"]*100,e["ci95_bootstrap"][0]*100,e["ci95_bootstrap"][1]*100,test)); labels.append(f"{tl}: {cl}")
y=np.arange(len(rows))[::-1]
for yi,(d,lo,hi,test) in zip(y,rows):
    c=BL if test=="ou_ss" else OR; mk="o" if test=="ou_ss" else "s"
    a1.plot([lo,hi],[yi,yi],color=c,lw=1.6); a1.plot(d,yi,mk,color=c,ms=6)
a1.axvline(0,color="#8a8f98",lw=0.8); a1.set_yticks(y); a1.set_yticklabels(labels,fontsize=8); a1.set_xlabel("Difference in detection (pp)")
a1.set_title("(a) Loss at twofold escalation",loc="left",fontsize=9); a1.grid(axis="x",color="#e3e6ea",lw=0.6); a1.set_axisbelow(True)
dr=A["dose_response"]["escalation|ou_ss|var"]; ms=[float(k) for k in dr["mean_by_m"]]; vals=[v*100 for v in dr["mean_by_m"].values()]
a2.plot(ms,vals,"-o",color=BL,lw=1.6,ms=5); a2.axhline(0,color="#8a8f98",lw=0.8); a2.set_xscale("log"); a2.set_xticks(ms); a2.set_xticklabels([f"{m:g}" for m in ms]); a2.minorticks_off()
a2.set_xlabel("Escalation factor m (log scale)"); a2.set_ylabel("Escalation component (pp)")
a2.set_title("(b) Escalation component by m",loc="left",fontsize=9); a2.grid(axis="y",color="#e3e6ea",lw=0.6); a2.set_axisbelow(True)
fig.tight_layout(); save(fig,"Figure_2"); plt.close(fig)
# ---------- Figure 3: MIMIC hourly profile (from frozen M1 summary)
M=json.load(open(os.path.join(BASE,"m1_mimic","results","mimic_M1_v122_full_summary.json")))["profile"]
c=np.array(M["cases_mean_draws_by_hour_before_event"]); k=np.array(M["controls_mean_draws_by_hour_before_event"]); x=-(np.arange(48)+0.5)
fig,ax=plt.subplots(figsize=(6.8,3.8)); ax.axvspan(-24,0,color="#f1f3f6",zorder=0); ax.axvline(-24,color="#8a8f98",lw=0.8,ls="--")
ax.plot(x,c,color=BL,lw=1.8,marker="o",ms=2.8,label="Ward-to-ICU cases"); ax.plot(x,k,color=OR,lw=1.8,marker="s",ms=2.6,label="Matched ward controls")
ax.set_xlim(-48,0); ax.set_ylim(0,0.7); ax.set_xticks(range(-48,1,6)); ax.set_xticklabels([str(abs(v)) for v in range(-48,1,6)])
ax.set_xlabel("Hours before ICU entry (cases) or matched index time (controls)"); ax.set_ylabel("Specimen draws per patient per hour")
ax.legend(frameon=False,loc="upper left"); ax.grid(axis="y",color="#e3e6ea",lw=0.6); ax.set_axisbelow(True)
fig.tight_layout(); save(fig,"Figure_3"); plt.close(fig)
# ---------- Figure S1: clock hour
cc=np.array(M["cases_share_of_draws_by_clock_hour"])*100; kc=np.array(M["controls_share_of_draws_by_clock_hour"])*100; h=np.arange(24)
fig,ax=plt.subplots(figsize=(6.8,3.4)); ax.bar(h-0.21,cc,0.4,color=BL,label="Ward-to-ICU cases"); ax.bar(h+0.21,kc,0.4,color=OR,label="Matched ward controls",hatch="//",edgecolor="white")
ax.set_xticks(range(0,24,2)); ax.set_xticklabels([f"{v:02d}:00" for v in range(0,24,2)],fontsize=8); ax.set_xlabel("Clock hour of collection"); ax.set_ylabel("Share of draws (%)"); ax.legend(frameon=False)
fig.tight_layout(); save(fig,"Figure_S1"); plt.close(fig)
print("ok")
