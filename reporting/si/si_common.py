# -*- coding: utf-8 -*-
"""Shared helpers for Supporting Information S2-S5 (python-docx).
Numbers are taken only from the frozen outputs; rounding is the exact stored decimal,
half away from zero (same rule as the manuscript registry)."""
import json, os
from decimal import Decimal, ROUND_HALF_UP
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

PAPER = ("Early-warning statistics under escalating observation density: a simulation study "
         "with evidence from hospital laboratory data")
AUTHOR = "Lakshmi Kalyani Chinthala"
ZEN = os.environ.get("SI_DOI", "the Zenodo record of this archive")
DEP = os.environ.get("SI_ARCHIVE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
OUT = os.environ.get("SI_OUT", ".")
FIG_S1 = os.environ.get("SI_FIG_S1", "Figure_S1.png")
A1F = DEP + "/a1_simulation/results/a1_summary_20261031_220543.json"
M1F = DEP + "/m1_mimic/results/mimic_M1_v122_full_summary.json"
M1F0 = DEP + "/m1_mimic/results/first_run_v1.2/mimic_M1_v1.2_first_run_summary.json"
A1J = DEP + "/a1_simulation/rebuild_v322/config/Simulation_Addendum_A1_v1.2_FROZEN.json"
P2J = DEP + "/a1_simulation/rebuild_v322/config/Simulation_Rebuild_Config_v3.2.2_Phase2.json"

USED = []   # every formatted number, for the reconciliation check


def _d(x):
    return Decimal(repr(float(x)))


def num(x, dp, sign=False):
    v = _d(x).quantize(Decimal(1).scaleb(-dp), rounding=ROUND_HALF_UP)
    s = str(v).replace("-", "−")
    if sign and v > 0:
        s = "+" + s
    if s in ("−0." + "0" * dp, "+0." + "0" * dp) and dp > 0:
        s = "0." + "0" * dp
    USED.append(s)
    return s


def pp(x, dp=1, sign=False):
    v = (_d(x) * 100).quantize(Decimal(1).scaleb(-dp), rounding=ROUND_HALF_UP)
    s = str(v).replace("-", "−")
    if sign and v > 0:
        s = "+" + s
    if s in ("−0.0", "+0.0"):
        s = "0.0"
    USED.append(s)
    return s


def ppci(c, dp=1):
    return f"{pp(c[0], dp)} to {pp(c[1], dp)}"


def pfmt(p, dp=3):
    if p is None:
        return "–"
    if p < 0.001:
        return "<0.001"
    return num(p, dp)


ARM = {"regular_N": "Even", "informative_reduced_N": "Uneven", "poisson_matched": "Random-time", "thinned": "Thinned"}
TEST = {"ou_ss": "Timing-aware", "ar1": "AR(1)", "shuffle": "Shuffle", "ar1_rep": "AR(1)", "shuffle_rep": "Shuffle"}
IND = {"var": "Variance", "ac": "Lag-1 AC"}
LAB = {"CALIBRATED": "Calibrated", "LOW": "Low", "HIGH_NOT_EXCLUDED": "High not excluded", "HIGH": "High"}
CLS = {"NO_RECOVERY": "No recovery", "GENUINE_RECOVERY": "Genuine recovery",
       "IMPROVEMENT_FA_CONTROL_NOT_ESTABLISHED": "Improvement, false-alarm control not established",
       "IMPROVEMENT_AT_FALSE_ALARM_COST": "Improvement at false-alarm cost"}
SCEN = {"F1": "Primary design (confirmatory cohort)", "XS1": "Independent trigger", "XR1": "Single crossing",
        "XR2": "Threshold 2.0 SD", "XR3": "Threshold 2.5 SD", "XR4": "Oracle trigger", "XR5": "m = 1.5",
        "XR6": "m = 3", "XR7": "m = 4", "XR8": "m = 6", "XR9": "Ramp", "XR10": "Smooth oracle schedule",
        "XS2": "Slowing with level drop"}
NULLB = {"G1a": "Stable slow, θ = 0.2, m = 2 (gate)", "G1b": "Stable slow, θ = 0.2, m = 6",
         "G1c": "Stable slow, θ = 0.05, m = 2", "G1d": "Stable slow, θ = 0.05, m = 6",
         "XN1": "Level drop without slowing", "XN2": "Stable, θ = 1.2",
         "P5FA": "Stable, θ = 1.2 (preprocessing)", "E2FA": "Stable, θ = 1.2 (continuous-time)"}


def rep_name(a):
    if a == "index":
        return "Raw series"
    if a.startswith("fixed_time|W"):
        return f"Fixed-time windows, width {a.split('W')[1]}"
    if a.startswith("binned|"):
        _, b, f = a.split("|")
        fill = {"linear_interp": "interpolated", "locf": "carried forward", "drop": "dropped"}[f]
        return f"{b} bins, empty {fill}"
    if a.startswith("interp|"):
        return f"Linear interpolation, {a.split('|')[1]} points"
    if a == "ct|W30":
        return "Continuous-time, width 30"
    return a


class Doc:
    def __init__(self, sx, title):
        self.d = Document()
        st = self.d.styles["Normal"]
        st.font.name = "Times New Roman"; st.font.size = Pt(12)
        st.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
        s = self.d.sections[0]
        s.top_margin = s.bottom_margin = Inches(1); s.left_margin = s.right_margin = Inches(1)
        self.tn = 0
        self.sx = sx
        self.para(f"Supporting Information {sx}. {title}", bold=True, size=15, align="c", after=10)
        self.para(f"Supporting Information for: {PAPER}. {AUTHOR}.", size=12, after=14)

    def para(self, text, bold=False, italic=False, size=12, align=None, after=8, keep=False):
        p = self.d.add_paragraph(); p.paragraph_format.space_after = Pt(after)
        p.paragraph_format.line_spacing = 1.15
        if align == "c":
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        if keep:
            p.paragraph_format.keep_with_next = True
        r = p.add_run(text); r.bold = bold; r.italic = italic; r.font.size = Pt(size)
        return p

    def head(self, text):
        p = self.para(text, bold=True, size=13, after=6, keep=True)
        p.paragraph_format.space_before = Pt(10)

    def bullets(self, items):
        for t in items:
            p = self.d.add_paragraph(style="List Bullet"); p.paragraph_format.space_after = Pt(3)
            r = p.add_run(t); r.font.size = Pt(12)

    def landscape(self, on):
        s = self.d.add_section()
        if on:
            s.orientation = WD_ORIENT.LANDSCAPE; s.page_width, s.page_height = Inches(11), Inches(8.5)
        else:
            s.orientation = WD_ORIENT.PORTRAIT; s.page_width, s.page_height = Inches(8.5), Inches(11)
        s.top_margin = s.bottom_margin = Inches(0.8); s.left_margin = s.right_margin = Inches(0.8)

    def table(self, title, header, rows, widths, note=None, size=8.5, group_rows=()):
        self.tn += 1
        cap = self.para(f"Table {self.sx}.{self.tn}. {title}", bold=True, size=11, after=4, keep=True)
        t = self.d.add_table(rows=1, cols=len(header)); t.style = "Table Grid"
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        t.autofit = False
        def fill(cells, vals, bold=False, shade=None):
            for i, (c, v) in enumerate(zip(cells, vals)):
                c.width = Inches(widths[i])
                c.text = ""
                pr = c.paragraphs[0]; pr.paragraph_format.space_after = Pt(0)
                r = pr.add_run(str(v)); r.font.size = Pt(size); r.bold = bold
                if shade:
                    tcPr = c._tc.get_or_add_tcPr(); sh = OxmlElement("w:shd")
                    sh.set(qn("w:val"), "clear"); sh.set(qn("w:color"), "auto"); sh.set(qn("w:fill"), shade); tcPr.append(sh)
        for i, col in enumerate(t.columns):
            col.width = Inches(widths[i])
        tblPr = t._tbl.tblPr
        tw = OxmlElement("w:tblW"); tw.set(qn("w:w"), str(int(sum(widths) * 1440))); tw.set(qn("w:type"), "dxa")
        for old in tblPr.findall(qn("w:tblW")):
            tblPr.remove(old)
        tblPr.append(tw)
        lay = OxmlElement("w:tblLayout"); lay.set(qn("w:type"), "fixed"); tblPr.append(lay)
        for gc, w in zip(t._tbl.tblGrid.findall(qn("w:gridCol")), widths):
            gc.set(qn("w:w"), str(int(w * 1440)))
        fill(t.rows[0].cells, header, bold=True, shade="E7E9EE")
        # repeat header row on each page
        trPr = t.rows[0]._tr.get_or_add_trPr(); th = OxmlElement("w:tblHeader"); th.set(qn("w:val"), "true"); trPr.append(th)
        for i, row in enumerate(rows):
            cells = t.add_row().cells
            if isinstance(row, str):          # group heading row
                m = cells[0].merge(cells[-1]); m.text = ""
                r = m.paragraphs[0].add_run(row); r.bold = True; r.font.size = Pt(size)
                m.paragraphs[0].paragraph_format.space_after = Pt(0)
                tcPr = m._tc.get_or_add_tcPr(); sh = OxmlElement("w:shd")
                sh.set(qn("w:val"), "clear"); sh.set(qn("w:color"), "auto"); sh.set(qn("w:fill"), "F4F5F7"); tcPr.append(sh)
            else:
                fill(cells, row)
        if note:
            self.para(note, italic=True, size=9.5, after=12)
        else:
            self.para("", size=4, after=8)

    def save(self, path):
        self.d.save(path)


def load(p):
    return json.load(open(p, encoding="utf-8"))
