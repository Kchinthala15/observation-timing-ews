"""Builds the manuscript number registry from the frozen outputs only.
A1: frozen summary a1_summary_20261031_220543 (A1 9aff4ef7...; config 0473d483...)
M1: mimic_M1_v122_full_summary.json (script 1b869172...)
Rounding: exact decimal of the stored value, half away from zero."""
import json, hashlib, os
from decimal import Decimal, ROUND_HALF_UP
BASE=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..")
A1F=os.path.join(BASE,"a1_simulation","results","a1_summary_20261031_220543.json"); M1F=os.path.join(BASE,"m1_mimic","results","mimic_M1_v122_full_summary.json")
A=json.load(open(A1F)); M=json.load(open(M1F))
R={}; SRC={}
def q(x,dp): return Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-dp),rounding=ROUND_HALF_UP)
def pp(x,dp=1,sign=False):
    v=q(Decimal(repr(float(x)))*100 if False else float(Decimal(repr(float(x)))*100),dp)
    s=f"{v}"; s=s.replace("-","−")
    if sign and v>0: s="+"+s
    return s
def ppx(x,dp=1,sign=False):
    v=(Decimal(repr(float(x)))*100).quantize(Decimal(1).scaleb(-dp),rounding=ROUND_HALF_UP)
    s=str(v).replace("-","−")
    if sign and v>0: s="+"+s
    if s in ("−0.0","+0.0"): s="0.0"
    return s
def num(x,dp,sign=False):
    v=Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-dp),rounding=ROUND_HALF_UP)
    s=str(v).replace("-","−")
    if sign and v>0: s="+"+s
    return s
def put(k,v,src): R[k]=v; SRC[k]=src
def ci(k,lo,hi,src,dp=1,f=ppx): put(k,f"{f(lo,dp)} to {f(hi,dp)}",src)
st={}
for fam in ("S_Q1_family","S_Q3_family","S_FA_family"):
    for e in A[fam]: st[e["bootstrap_stream"]]=e
# ---- calibration / gate
cal=A["calibration"]
g=[cal[f"G1a|{a}|index|ou_ss|{i}"]["rate"] for a in ("regular_N","informative_reduced_N","poisson_matched") for i in ("var","ac")]
put("g1a_ouss_min",ppx(min(g)),"calibration G1a ou_ss"); put("g1a_ouss_max",ppx(max(g)),"calibration G1a ou_ss")
c=cal["G1a|informative_reduced_N|index|ar1|var"]; put("g1a_ar1_unev_var",ppx(c["rate"]),"calibration"); put("g1a_ar1_unev_var_label",c["label"],"calibration")
labs={}
for k,v in cal.items():
    labs.setdefault(v["label"],0); labs[v["label"]]+=1
put("cal_cells_total",str(len(cal)),"calibration"); 
for L,n in labs.items(): put(f"cal_n_{L}",str(n),"calibration")
# ---- Q1 F1
for t in ("ou_ss","ar1","shuffle"):
  for i in ("var","ac"):
    for comp,lab in (("total(inf-reg)","tot"),("irregularity(pm-reg)","irr"),("escalation(inf-pm)","esc")):
        e=st[f"bootstrap|ou_csd|A1|F1|Q1|{comp}|{t}|{i}"]
        put(f"F1_{lab}_{t}_{i}",ppx(e["diff"],sign=True),e["bootstrap_stream"]); ci(f"F1_{lab}_{t}_{i}_ci",*e["ci95_bootstrap"],e["bootstrap_stream"])
        if e.get("holm_p") is not None: put(f"F1_{lab}_{t}_{i}_holm",("<0.001" if e["holm_p"]<0.001 else num(e["holm_p"],2)),e["bootstrap_stream"])
e=st["bootstrap|ou_csd|A1|F1|Q1|total(inf-reg)|ou_ss|var"]
put("F1_det_inf_ouss_var",ppx(e["rate_A"]),"F1"); put("F1_det_reg_ouss_var",ppx(e["rate_B"]),"F1")
e=st["bootstrap|ou_csd|A1|F1|Q1|total(inf-reg)|ou_ss|ac"]
put("F1_det_inf_ouss_ac",ppx(e["rate_A"]),"F1"); put("F1_det_reg_ouss_ac",ppx(e["rate_B"]),"F1")
# ---- confirmatory H1/H2
cf=A["confirmatory_F1"]
v=cf["ar1"]["var"]
put("H1_raw",ppx(v["H1"]["diff"],sign=True),"confirmatory_F1 ar1 var H1"); ci("H1_raw_ci",*v["H1"]["ci97.5_bootstrap"],"confirmatory_F1")
put("H1_det_ouss",ppx(v["H1"]["rate_A"]),"F1"); put("H1_det_ar1",ppx(v["H1"]["rate_B"]),"F1")
put("H2_raw",ppx(v["H2"]["diff_in_diff"],sign=True),"confirmatory_F1"); ci("H2_raw_ci",*v["H2"]["ci97.5_bootstrap"],"confirmatory_F1")
for ref in ("G1a","XN2"):
    s=v["size_adjusted"][ref]
    put(f"H1_adj_{ref}",ppx(s["H1"]["diff"],sign=True),f"size_adjusted {ref}"); ci(f"H1_adj_{ref}_ci",*s["H1"]["ci97.5_bootstrap"],"size_adjusted")
    put(f"H2_adj_{ref}",ppx(s["H2"]["diff_in_diff"],sign=True),f"size_adjusted {ref}"); ci(f"H2_adj_{ref}_ci",*s["H2"]["ci97.5_bootstrap"],"size_adjusted")
    for kk,val in s["alpha_star"].items(): put(f"astar_{ref}_{kk.replace('|','_')}",num(val,4),"alpha_star")
a=cf["ar1"]["ac"]
put("AC_H1_raw",ppx(a["H1"]["diff"],sign=True),"ar1 ac H1"); ci("AC_H1_raw_ci",*a["H1"]["ci95_bootstrap"],"ar1 ac")
put("AC_H1_adj_G1a",ppx(a["size_adjusted"]["G1a"]["H1"]["diff"],sign=True),"ar1 ac adj"); ci("AC_H1_adj_G1a_ci",*a["size_adjusted"]["G1a"]["H1"]["ci95_bootstrap"],"ar1 ac adj")
put("AC_H1_adj_XN2",ppx(a["size_adjusted"]["XN2"]["H1"]["diff"],sign=True),"ar1 ac adj XN2"); ci("AC_H1_adj_XN2_ci",*a["size_adjusted"]["XN2"]["H1"]["ci95_bootstrap"],"ar1 ac adj XN2")
sv=cf["shuffle"]["var"]
put("SH_H1_raw",ppx(sv["H1"]["diff"],sign=True),"shuffle var"); ci("SH_H1_raw_ci",*sv["H1"]["ci95_bootstrap"],"shuffle var")
put("SH_H1_adj_G1a",ppx(sv["size_adjusted"]["G1a"]["H1"]["diff"],sign=True),"shuffle adj"); ci("SH_H1_adj_G1a_ci",*sv["size_adjusted"]["G1a"]["H1"]["ci95_bootstrap"],"shuffle adj")
# ---- scenarios
gains=[]
for e in A["S_Q3_family"]:
    s=e["bootstrap_stream"]
    if s.endswith("|Q3|gain|ar1|var"): gains.append((s.split("|")[3],e["diff"],e["ci95_bootstrap"]))
put("n_scen",str(len(gains)),"S_Q3 gains incl F1"); put("scen_gain_min",ppx(min(g[1] for g in gains),sign=True),"S_Q3"); put("scen_gain_max",ppx(max(g[1] for g in gains),sign=True),"S_Q3")
put("scen_gain_minlo",ppx(min(g[2][0] for g in gains)),"S_Q3")
SCEN={b:(ppx(dv,sign=True),f"{ppx(c_[0])} to {ppx(c_[1])}") for b,dv,c_ in gains}
put("XS2_gain",ppx(A["XS2_restriction_rule"]["gain_point_estimate"],sign=True),"XS2 rule")
# ---- state dependence and dose
sd=A["state_dependence_rule"]["ou_ss"]["var"]
put("SD_diff",ppx(sd["difference"],sign=True),"state_dependence"); ci("SD_ci",*sd["ci95"],"state_dependence")
put("SD_F1esc",ppx(sd["F1_escalation"],sign=True),"sd"); put("SD_XS1esc",ppx(sd["XS1_escalation"],sign=True),"sd")
dr=A["dose_response"]
for k in ("total|ou_ss|var","escalation|ou_ss|var","total|ou_ss|ac","escalation|ou_ss|ac"):
    kk=k.replace("|","_"); put(f"dose_{kk}",ppx(dr[k]["mean_per_patient_slope_per_log_m"],sign=True),"dose_response"); ci(f"dose_{kk}_ci",*dr[k]["ci95"],"dose_response")
    for m,val in dr[k]["mean_by_m"].items(): put(f"dose_{kk}_m{m.replace('.','p')}",ppx(val,sign=True),"dose_response")
# ---- P5
P5={}
for e in A["S_Q2_P5_family"]:
    P5[(e["analysis"],e["test"],e["indicator"])]=e
h2h={(e["analysis"],e["indicator"]):e for e in A["P5_head_to_head"]}
cls={}
for (an,te,ind),e in P5.items():
    cls.setdefault((te,ind),{}); c_=e.get("class"); cls[(te,ind)][c_]=cls[(te,ind)].get(c_,0)+1
json.dump({f"{k[0]}|{k[1]}":v for k,v in cls.items()},open("p5_class_counts.json","w"),indent=1)
put("P5_keys_example",str(list(P5.values())[0].keys()),"P5")
# ---- E2 / E2FA
for e in A["S_Q2_E2"]:
    ind=e["indicator"]
    for part in ("recovery","gap_even_benchmark","gap_ct_even","vs_even_index_ouss","remaining"):
        if part in e:
            put(f"E2_{part}_{ind}",ppx(e[part]["diff"],sign=True),"S_Q2_E2"); ci(f"E2_{part}_{ind}_ci",*e[part]["ci95_bootstrap"],"S_Q2_E2")
    put(f"E2_keys_{ind}",str(list(e.keys())),"E2")
    r=e["recovery"]; put(f"E2_det_ct_unev_{ind}",ppx(r["rate_A"]),"E2"); put(f"E2_det_ouss_unev_{ind}",ppx(r["rate_B"]),"E2")
for k,v in cal.items():
    if k.startswith("E2FA"): put("cal_"+k.replace("|","_"),ppx(v["rate"])+"|"+v["label"],"calibration")
    if k.startswith("XN1") and "|ac" in k: put("cal_"+k.replace("|","_"),ppx(v["rate"])+"|"+v["label"]+f"|{ppx(v['cp90'][0])}-{ppx(v['cp90'][1])}","calibration")
# ---- fold
fs=A["fold_selection"]["grid_results"]
for gr in fs: put(f"fold_{gr['sigma_f']}_{gr['r_end']}",f"{gr['tipped']}/{gr['n_enrolled']}|{ppx(gr['regular_N_variance_detection'])}","fold_selection")
# ---- M1
p=M["primary_24h"]["draws"]
put("M1_cases",f"{M['cohort']['n_ward_escalations']:,}","M1"); put("M1_controls",f"{M['cohort']['n_controls']:,}","M1")
put("M1_did",num(p["did_per_h"],3,sign=True),"M1"); put("M1_did_ci",f"{num(p['did_ci95'][0],3)} to {num(p['did_ci95'][1],3)}","M1")
put("M1_case_base",num(p["case_base_per_h"],3),"M1"); put("M1_case_pre",num(p["case_pre_per_h"],3),"M1")
put("M1_ctrl_base",num(p["control_base_per_h"],3),"M1"); put("M1_ctrl_pre",num(p["control_pre_per_h"],3),"M1")
put("M1_ror",num(p["ratio_of_ratios"],2),"M1"); put("M1_ror_ci",f"{num(p['ratio_of_ratios_ci95'][0],2)} to {num(p['ratio_of_ratios_ci95'][1],2)}","M1")
s12=M["sensitivity_12h"]["draws"]; put("M1_did12",num(s12["did_per_h"],3,sign=True),"M1"); put("M1_did12_ci",f"{num(s12['did_ci95'][0],3)} to {num(s12['did_ci95'][1],3)}","M1")
put("M1_candidates",f"{M['matching']['candidate_control_admissions']:,}","M1")
json.dump({"registry":R,"source":SRC,"inputs":{"A1_summary_sha256":hashlib.sha256(open(A1F,'rb').read()).hexdigest(),"M1_summary_sha256":hashlib.sha256(open(M1F,'rb').read()).hexdigest()},"scenarios":SCEN},open("registry.json","w"),indent=1,ensure_ascii=False)
print(len(R),"entries")
# ===== additions: P5 rows, E2 gaps
NAMES={"fixed_time|W20":"Fixed-time windows, width 20","fixed_time|W30":"Fixed-time windows, width 30","fixed_time|W40":"Fixed-time windows, width 40",
 "binned|30|linear_interp":"30 bins, interpolate empty","binned|30|locf":"30 bins, carry forward","binned|30|drop":"30 bins, drop empty",
 "binned|60|linear_interp":"60 bins, interpolate empty","binned|60|locf":"60 bins, carry forward","binned|60|drop":"60 bins, drop empty",
 "binned|120|linear_interp":"120 bins, interpolate empty","binned|120|locf":"120 bins, carry forward","binned|120|drop":"120 bins, drop empty",
 "interp|60":"Linear interpolation, 60-point grid"}
rows=[]
for (an,te,ind),e in P5.items():
    if ind!="var" or te!="ar1_rep": continue
    h=h2h[(an,ind)]["head_to_head_vs_index_ou_ss"]
    rows.append(dict(key=an,name=NAMES.get(an,an),det=ppx(e["recovery"]["rate_A"]),rec=ppx(e["recovery"]["diff"],sign=True),rec_ci=f"{ppx(e['recovery']['ci95_bootstrap'][0])} to {ppx(e['recovery']['ci95_bootstrap'][1])}",
       h2h=ppx(h["diff"],sign=True),h2h_ci=f"{ppx(h['ci95_bootstrap'][0])} to {ppx(h['ci95_bootstrap'][1])}",
       ir=ppx(e["informative_minus_regular"]["diff"],sign=True),cls=e["class"],holm=("<0.001" if e["holm_p"]<0.001 else num(e["holm_p"],3)),fa_i=e["fa_label_informative"],fa_r=e["fa_label_regular"]))
e0=list(P5.values())[0]; put("P5_ref_ar1_index_unev",ppx(e0["recovery"]["rate_B"]),"P5 recovery rate_B (index ar1)")
h0=list(h2h.values())[0]["head_to_head_vs_index_ou_ss"]; put("P5_ref_ouss_index_unev",ppx(h0["rate_B"]),"P5 h2h rate_B")
put("P5_ref_ouss_index_even",ppx(e0["gap_to_even_index_ou_ss"]["rate_B"]),"P5 gap rate_B")
irs=[]; 
for (an,te,ind),e in P5.items():
    if ind=="var": irs.append((te,e["informative_minus_regular"]["diff"]))
put("P5_ir_min_ar1",ppx(max(d for t,d in irs if t=="ar1_rep"),sign=True),"P5 inf-reg"); put("P5_ir_max_ar1",ppx(min(d for t,d in irs if t=="ar1_rep"),sign=True),"P5 inf-reg")
put("P5_ir_min_sh",ppx(max(d for t,d in irs if t=="shuffle_rep"),sign=True),"P5 inf-reg"); put("P5_ir_max_sh",ppx(min(d for t,d in irs if t=="shuffle_rep"),sign=True),"P5 inf-reg")
for te in ("ar1_rep","shuffle_rep"):
  for ind in ("var","ac"):
    for c_,n_ in cls[(te,ind)].items(): put(f"P5_cls_{te}_{ind}_{c_}",str(n_),"P5 classes")
fl={}
for k,v in cal.items():
    if k.startswith("P5FA"): fl[v["label"]]=fl.get(v["label"],0)+1
for L,n_ in fl.items(): put(f"P5FA_n_{L}",str(n_),"calibration P5FA")
put("P5FA_total",str(sum(fl.values())),"calibration P5FA")
for e in A["S_Q2_E2"]:
    ind=e["indicator"]
    put(f"E2_gapbench_{ind}",ppx(e["gap_to_even_index_ou_ss"]["diff"],sign=True),"E2"); ci(f"E2_gapbench_{ind}_ci",*e["gap_to_even_index_ou_ss"]["ci95_bootstrap"],"E2")
    put(f"E2_ir_{ind}",ppx(e["informative_minus_regular"]["diff"],sign=True),"E2"); ci(f"E2_ir_{ind}_ci",*e["informative_minus_regular"]["ci95_bootstrap"],"E2")
    put(f"E2_det_ouss_even_{ind}",ppx(e["gap_to_even_index_ou_ss"]["rate_B"]),"E2"); put(f"E2_det_ct_even_{ind}",ppx(e["informative_minus_regular"]["rate_B"]),"E2")
    put(f"E2_holm_{ind}",num(e["holm_p_joint_P5_E2"],4),"E2")
pp5=A["S_Q2_P5_family"]
json.dump({"registry":R,"source":SRC,"inputs":{"A1_summary_sha256":hashlib.sha256(open(A1F,'rb').read()).hexdigest(),"M1_summary_sha256":hashlib.sha256(open(M1F,'rb').read()).hexdigest()},"scenarios":SCEN,"p5_rows":rows},open("registry.json","w"),indent=1,ensure_ascii=False)
print(len(R),"entries total")
