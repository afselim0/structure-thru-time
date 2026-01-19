#!/usr/bin/env python3
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

from scipy.spatial import ConvexHull
from scipy.ndimage import gaussian_filter
from skimage.measure import marching_cubes
from matplotlib.path import Path as MplPath
# ----------------------- tunables ------------------------------------------
sample_line_width = 2
sample_line_alpha = 0.5

median_line_width = 5
median_line_alpha = 1.0

# Volume and surface parameters
central_percentile = 0.95
grid_res_xy = 256
gauss_sigma = 1.5
tube_alpha = 0.18
# ---------------------------------------------------------------------------
class PlotEmbeddings:
    def __init__(self, 
                 embeddings_df: pd.DataFrame,
                 labels_df: pd.DataFrame = None,
                 colors_df: pd.DataFrame = None,
                 generation_time: int = None
                 ):
        
        # Loading population labels
        if labels_df is None:
            embeddings_df["population"] = "unlabelled_pop"
            colors_df = None # Set colors_df to None to trigger default colors if no labels provided
        else:
            embeddings_df = embeddings_df.merge(
                labels_df[["sample_id","population"]],
                on='sample_id',
                how='left'
            )
            embeddings_df["population"] = embeddings_df["population"].fillna("unlabelled_pop")

        self.populations = embeddings_df["population"].unique()
        self.num_populations = embeddings_df["population"].nunique()

        # Loading population colors
        if colors_df is None:
            colors = px.colors.qualitative.Safe # A colorblind-friendly palette if no colors assigned
            colors_dict = {pop:colors[i%len(colors)] for i,pop in enumerate(self.populations)}
            embeddings_df["color"] = embeddings_df["population"].map(colors_dict)
        else:
            embeddings_df = embeddings_df.merge(
                colors_df[["population","color"]],
                on='population',
                how='left'
            )
            embeddings_df["color"] = embeddings_df["color"].fillna("#808080") # Default gray for missing colors
            unique_colors = embeddings_df[["population","color"]].drop_duplicates()
            colors_dict = {row["population"]:row["color"] for _,row in unique_colors.iterrows()}
            
        self.colors_dict = colors_dict
        if generation_time is not None:
            embeddings_df["time"] = embeddings_df["time"] * generation_time
        
        self.times = embeddings_df["time"].unique()
        self.generation_time = generation_time
        self.embeddings_df = embeddings_df
        self.fig = None
        

    def init_plot(self, time_scale: str = "log"):
        self.fig = go.Figure()
        
        # Setting axis labels
        x_label = "X1"
        y_label = "X2"
        z_label = "Time (generations ago)" if self.generation_time is None else "Time (years ago)"

        # Setting axis styles
        axis_style = dict(
            showline=True,
            showgrid=True,
            showbackground=True,
            linecolor="black",
            linewidth=1,
            ticks="outside",
            tickcolor="black",
            gridcolor="white",
            backgroundcolor="whitesmoke",
            tickmode="array"
        )
        
        # Setting equal scaling for X and Y axes
        x_min = self.embeddings_df["X1"].min()
        x_max = self.embeddings_df["X1"].max()
        y_min = self.embeddings_df["X2"].min()
        y_max = self.embeddings_df["X2"].max()

        common_min = min(x_min, y_min)
        common_max = max(x_max, y_max)
        pad = 0.05 * (common_max - common_min)
        common_min -= pad
        common_max += pad

        x_range = [common_min, common_max]
        y_range = [common_min, common_max]

        # Setting gridlines and ticks for time axis
        min_time = self.embeddings_df["time"].min()
        min_time = 1 if min_time <= 0 else min_time
        max_time = self.embeddings_df["time"].max()
        z_min_decade = int(np.floor(np.log10(min_time)))
        z_max_decade = int(np.ceil(np.log10(max_time)))
        z_ticks = 10 ** np.arange(z_min_decade, z_max_decade + 1)

        self.fig.update_layout(
            scene=dict(
                xaxis=dict(title=x_label, range=x_range, zeroline=True, **axis_style),
                yaxis=dict(title=y_label, range=y_range, zeroline=True, **axis_style),
                zaxis=dict(title=z_label, tickvals = z_ticks, ticktext = [str(int(z)) for z in z_ticks], **axis_style, type=time_scale),
                aspectmode="manual",
                aspectratio=dict(x=1, y=1, z=1)
            )
        )

    def add_legend(self):
        for pop in self.populations:
            color = self.colors_dict.get(pop, "#808080")

            self.fig.add_trace(
                go.Scatter3d(
                    x=[None],
                    y=[None],
                    z=[None],
                    mode="markers",
                    marker=dict(size=10, color=color),
                    name=pop,
                    legendgroup=pop,
                    showlegend=True
                )
            )

    def save_static(self,filename, format="png", width=600, height=600, scale=2, eye=dict(x=1.8, y=1.8, z=0.3), up=dict(x=0, y=0, z=1)):
        self.fig.update_layout(scene=dict(camera=dict(eye=eye, up=up)))
        pio.write_image(fig = self.fig, file = f"{filename}.{format}",format=format,width=width,height=height,scale=scale)

    def add_sample_trajectories(self,line_width: int = sample_line_width,alpha: float = sample_line_alpha):
        samples = self.embeddings_df["sample_id"].unique()
        for sample in samples:
            sample_data = self.embeddings_df[self.embeddings_df["sample_id"] == sample]
            population = sample_data["population"].values[0]
            color = self.colors_dict.get(population,'#888888')
            self.fig.add_trace(
                go.Scatter3d(
                    x=sample_data["X1"],
                    y=sample_data["X2"],
                    z=sample_data["time"],
                    mode='lines',
                    line=dict(color=color, width=line_width),
                    opacity=alpha,
                    name=f'{population} - {sample}',
                    hovertemplate=f"{population}: {sample}<br>Time: %{{z:.2f}}<extra></extra>",
                    showlegend=False,
                    legendgroup=population
                )
            )

    def add_population_medians(self,line_width: int = median_line_width, alpha: float = median_line_alpha):
        median_df = (
            self.embeddings_df
            .groupby(["population", "time"], as_index=False)
            .agg(
                X1=("X1", "median"),
                X2=("X2", "median"),
                color=("color", "first"),
            )
        )

        for population in self.populations:
            pop_data = median_df[median_df["population"] == population]
            color = self.colors_dict.get(population, "#808080")
            self.fig.add_trace(
                go.Scatter3d(
                    x=pop_data["X1"],
                    y=pop_data["X2"],
                    z=pop_data["time"],
                    mode='lines',
                    line=dict(color=color, width=line_width),
                    name=f'{population} Median',
                    hovertemplate=f"{population} Median<br>Time: %{{z:.2f}}<extra></extra>",
                    showlegend=True,
                    legendgroup=population,
                    opacity=alpha
                )
            )

    def add_convex_hull(self):
        for population in self.populations:
            color = self.colors_dict.get(population, "#808080")
            vol_data=build_volume(self.embeddings_df,population,grid_res_xy)
            vol,xs,ys,ts=vol_data
            surf=surface_from_volume(vol,xs,ys,ts,gauss_sigma)
            if surf is not None:
                verts,faces=surf
                i,j,k=faces.T
                self.fig.add_trace(go.Mesh3d(x=verts[:,0],y=verts[:,1],z=verts[:,2],i=i,j=j,k=k,
                                         color=color,opacity=tube_alpha,name=population,legendgroup=population,showlegend=False,hoverinfo='skip'))

def slice_convex_hull(points: np.ndarray) -> np.ndarray:
    # Computes the convex hull for a given population at a given time point
    # points: Nx2 array of (x,y) coordinates of samples belonging to the population at that time point
    if len(points)<3: return points
    center=points.mean(axis=0)
    d=np.linalg.norm(points-center,axis=1)
    keep=points[d<=np.percentile(d, central_percentile*100)]
    if len(keep)<3: return keep
    hull=ConvexHull(keep)
    return keep[hull.vertices]

def rasterise_polygon(poly: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    if len(poly)<3: return np.zeros((len(ys),len(xs)),dtype=bool)
    path=MplPath(poly)
    xv,yv=np.meshgrid(xs,ys)
    pts=np.vstack([xv.ravel(),yv.ravel()]).T
    return path.contains_points(pts).reshape(len(ys),len(xs))

def build_volume(df: pd.DataFrame,pop:str,grid_res:int):
    sub=df[df.population==pop]
    xs_all,ys_all=sub['X1'],sub['X2']
    pad_x=0.05*(xs_all.max()-xs_all.min())
    pad_y=0.05*(ys_all.max()-ys_all.min())
    xs=np.linspace(xs_all.min()-pad_x,xs_all.max()+pad_x,grid_res)
    ys=np.linspace(ys_all.min()-pad_y,ys_all.max()+pad_y,grid_res)
    times=np.sort(sub.time.unique())
    ts = times
    vol=np.zeros((len(ts),grid_res,grid_res),dtype=bool)
    for i,t in enumerate(times): vol[i]=rasterise_polygon(slice_convex_hull(sub[sub.time==t][['X1','X2']].values),xs,ys)
    return vol,xs,ys,ts

def surface_from_volume(vol:np.ndarray,xs:np.ndarray,ys:np.ndarray,ts:np.ndarray,sigma:float):
    blurred=gaussian_filter(vol.astype(float),sigma=sigma)
    lo,hi=blurred.min(),blurred.max()
    if hi<=lo: print("Empty volume; skipping iso-surface (%.3f–%.3f)",lo,hi); return None
    level=0.5
    if not(lo<level<hi): level=lo+0.5*(hi-lo)#; print("Adjusted iso level → %.3f (%.3f–%.3f)",level,lo,hi)
    try: verts,faces,_,_=marching_cubes(blurred,level=level)
    except RuntimeError as e: print("No surface found: %s; skipping",e); return None
    nx,ny,nz=len(xs)-1,len(ys)-1,len(ts)-1
    vx=xs[0]+verts[:,2]/nx*(xs[-1]-xs[0]); vy=ys[0]+verts[:,1]/ny*(ys[-1]-ys[0]); vz=ts[0]+verts[:,0]/nz*(ts[-1]-ts[0])
    return np.stack([vx,vy,vz],axis=1),faces