#%%
"""
genetic_population_palette.py — hierarchical palette generator **plus** legend plotter with adjustable spread
===========================================================================

• Pops in 1000 Genomes use their original hex codes (fixed anchors).
• New pops are generated around these anchors with controllable hue span.

Key tunable parameters:
  --hue-span   : maximum ± degrees to jitter hue around anchor (default: 8°)
  --sat-range  : saturation range as min,max (default: auto from anchor HSLuv)
  --lum-range  : lightness range as min,max (default: auto from anchor HSLuv)

Imputation beyond fixed anchors:
  If you need to fill more slots than anchors provide, the script evenly allocates
  new pops across anchors and uses jitter to spread them––in effect extrapolating
  the palette direction by offsetting from each anchor into its neighbourhood.

------------------------------------------------------------
CLI
------------------------------------------------------------
$ python genetic_population_palette.py \
        --input      populations.csv \
        --output     population_colours.csv \
        --plot       palette_legend.png      # optional
        --hue-span   12                    # wider hue spread
        --sat-range  50,90                # saturation between 50% and 90%
        --lum-range  40,80                # lightness between 40% and 80%

Dependencies:
    pandas numpy matplotlib hsluv
"""
from __future__ import annotations
import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np

# ------------------------------------------------------------------
# Hard‑coded 1000 Genomes colours (fixed anchors)
# ------------------------------------------------------------------
HARD_CODED_COLORS: Dict[str, str] = {
    "ESN": "#F2C854", "GWD": "#E1B950", "LWK": "#B59246", "MSL": "#C9B24F", "YRI": "#EAB14F",
    "ACB": "#E1954A", "ASW": "#D76E3B", "CLM": "#AD4A44", "MXL": "#BA393F", "PEL": "#C9483D", "PUR": "#B04634",
    "CDX": "#4C9D49", "CHB": "#A3CA4D", "CHS": "#7BAB59", "JPT": "#3C884A", "KHV": "#62AC5A",
    "CEU": "#2D4C85", "FIN": "#62B0BF", "GBR": "#77BED1", "IBS": "#6984B5", "TSI": "#293260",
    "BEB": "#6D2D79", "GIH": "#5A408D", "ITU": "#963B5A", "PJL": "#BF3B7D", "STU": "#87568B",
}

# ------------------------------------------------------------------
# Colour conversion and hex ↔ HSLuv / HLS helpers
# ------------------------------------------------------------------
try:
    from hsluv import rgb_to_hsluv, hsluv_to_rgb
    def hex_to_hsluv(hexcol: str) -> Tuple[float,float,float]:
        r,g,b = [int(hexcol[i:i+2],16)/255.0 for i in (1,3,5)]
        return rgb_to_hsluv([r,g,b])    # [H,S,L]
    def hsluv_to_hex(h: float, s: float, l: float) -> str:
        r,g,b = hsluv_to_rgb([h,s,l])
        return "#" + "".join(f"{int(round(c*255)):02X}" for c in (r,g,b))
except ModuleNotFoundError:
    import colorsys
    def hex_to_hsluv(hexcol: str) -> Tuple[float,float,float]:
        r,g,b = [int(hexcol[i:i+2],16)/255.0 for i in (1,3,5)]
        h,l,s = colorsys.rgb_to_hls(r,g,b)
        return (h*360, s*100, l*100)
    def hsluv_to_hex(h: float, s: float, l: float) -> str:
        r,g,b = colorsys.hls_to_rgb(h/360, l/100, s/100)
        return "#" + "".join(f"{int(round(c*255)):02X}" for c in (r,g,b))

# ------------------------------------------------------------------
# Anchor hues for each region (fallback)
# ------------------------------------------------------------------
ANCHOR_HUES: Dict[str, float] = {
    "Africa": 75.0, "America": 15.0, "Ancient": 0.0,
    "Central_Asia_Siberia": 200.0, "Central_South_Asia": 310.0,
    "East_Asia": 115.0, "Middle_East": 45.0,
    "Oceania": 220.0, "South_Asia": 290.0,
    "West_Eurasia": 260.0, "Western_Europe": 240.0,
}

# ------------------------------------------------------------------
# Generate palette around anchors with adjustable parameters
# ------------------------------------------------------------------

def jitter_palette(base: Tuple[float,float,float], k: int,
                   hue_span: float, sat_range: Tuple[float,float],
                   lum_range: Tuple[float,float]) -> List[str]:
    """
    Generate k colours by jittering HSL around base:
      - H ∈ base_H ± hue_span (biggest |offset| first)
      - S ∈ linspace(sat_max→sat_min)
      - L ∈ linspace(lum_min→lum_max)
    """
    h0,s0,l0 = base
    if k == 0:
        return []
    # H offsets
    offsets = sorted(np.linspace(-hue_span, hue_span, k), key=lambda x: -abs(x))
    # S/L ranges
    sm, sM = sat_range
    lm, lM = lum_range
    s_vals = np.linspace(sM, sm, k)
    l_vals = np.linspace(lm, lM, k)
    return [hsluv_to_hex((h0+dh)%360, s, l)
            for dh, s, l in zip(offsets, s_vals, l_vals)]

# ------------------------------------------------------------------
# Build full palette, seeds + generated
# ------------------------------------------------------------------

def build_palette(df: pd.DataFrame, hue_span: float,
                  sat_range: Tuple[float,float], lum_range: Tuple[float,float]) -> pd.DataFrame:
    unknown = set(df.region) - set(ANCHOR_HUES)
    if unknown:
        raise ValueError(f"Unknown region(s): {', '.join(sorted(unknown))}")
    rows: List[pd.DataFrame] = []
    for region, sub in df.groupby("region"):
        sorted_sub = sub.sort_values("n", ascending=False).reset_index(drop=True)
        fixed = [(pop,HARD_CODED_COLORS[pop])
                 for pop in sorted_sub.population if pop in HARD_CODED_COLORS]
        anchor_hsl = [hex_to_hsluv(hex_) for _,hex_ in fixed]
        others = [pop for pop in sorted_sub.population if pop not in HARD_CODED_COLORS]
        k = len(others)
        if anchor_hsl:
            per_anchor = math.ceil(k/len(anchor_hsl))
            pal = []
            for base in anchor_hsl:
                pal.extend(jitter_palette(base, per_anchor, hue_span, sat_range, lum_range))
            pal = pal[:k]
        else:
            base = (ANCHOR_HUES[region], *[(s+e)/2 for s,e in (sat_range,lum_range)])
            pal = jitter_palette(base, k, hue_span, sat_range, lum_range)
        # merge hex
        hex_list=[]; idx=0
        for pop in sorted_sub.population:
            if pop in HARD_CODED_COLORS:
                hex_list.append(HARD_CODED_COLORS[pop])
            else:
                hex_list.append(pal[idx]); idx+=1
        rows.append(sorted_sub.assign(hex_colour=hex_list))
    return pd.concat(rows, ignore_index=True)

# ------------------------------------------------------------------
# Legend plot (optional)
# ------------------------------------------------------------------

def draw_palette(df: pd.DataFrame, plot_path: Path):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    regions = list(ANCHOR_HUES)
    maxc = df.groupby("region").size().max()
    fig, ax = plt.subplots(figsize=(maxc*0.45+3, len(regions)*0.6+1))
    ax.set_xlim(0, maxc+1); ax.set_ylim(0, len(regions)); ax.axis('off')
    for ridx, region in enumerate(reversed(regions)):
        y=len(regions)-1-ridx
        sub=df[df.region==region].reset_index(drop=True)
        ax.text(-0.5, y+0.5, region.replace('_',' '), ha='right', va='center', fontweight='bold', fontsize = 28)
        for x,(pop,col) in enumerate(zip(sub.population, sub.hex_colour)):
            ax.add_patch(Rectangle((x,y),1,1,facecolor=col,edgecolor='black',linewidth=0.4))
            ax.text(x+0.5, y+0.5, pop, ha='center', va='center', fontsize=6,
                    color='white' if _is_dark(col) else 'black')
    plt.tight_layout(); fig.savefig(plot_path, dpi=600, bbox_inches='tight'); print(f"Saved → {plot_path}")

def _is_dark(hexc: str) -> bool:
    r,g,b = [int(hexc[i:i+2],16) for i in (1,3,5)]
    return 0.299*r+0.587*g+0.114*b < 128

# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def parse_args(argv: list[str]) -> argparse.Namespace:
    p=argparse.ArgumentParser()
    p.add_argument('-i','--input', required=True)
    p.add_argument('-o','--output',required=True)
    p.add_argument('-p','--plot', help='legend file')
    p.add_argument('--hue-span', type=float, default=8.0,
                   help='± degrees around anchor hue')
    p.add_argument('--sat-range', default='55,85',
                   help='saturation min,max (comma-separated)')
    p.add_argument('--lum-range', default='45,70',
                   help='lightness min,max')
    return p.parse_args(argv)


def main(argv: list[str]|None=None):
    args=parse_args(argv or sys.argv[1:])
    df=pd.read_csv(args.input)
    for c in ('population','region','n'):
        if c not in df.columns: raise ValueError(f'Missing {c}')
    sat=tuple(map(float,args.sat_range.split(',')))
    lum=tuple(map(float,args.lum_range.split(',')))
    pal=build_palette(df, args.hue_span, sat, lum)
    pal.to_csv(args.output,index=False)
    print(f"Wrote {len(pal)} → {args.output}")
    if args.plot: draw_palette(pal, Path(args.plot))
#%%
if __name__=='__main__': main()

# %%
df = pd.read_csv("metadata_n.csv")
for c in ('population','region','n'):
    if c not in df.columns: raise ValueError(f'Missing {c}')
sat_range = '50,90'
lum_range = '40,80'
hue_span = 12.0
sat=tuple(map(float,sat_range.split(',')))
lum=tuple(map(float,lum_range.split(',')))
pal=build_palette(df, hue_span, sat, lum)
draw_palette(pal, Path("legend.png"))
# %%
