#%%
import math
import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.colors as mcolors
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go

# ----------------------- tunables ------------------------------------------
CENTRAL_PERCENTILE    = 0.95
GRID_RES_XY           = 256
GAUSS_SIGMA           = 1.5
TUBE_ALPHA            = 0.18
INDIVIDUAL_LINE_ALPHA = 0.10
MEDIAN_SMOOTH_WINDOW  = 5
MEDIAN_LINE_WIDTH     = 5
# ---------------------------------------------------------------------------
embeddings_df = pd.read_parquet("../results/empirical/wohns/test/embeddings.parquet")
labels_df = pd.read_csv("../data/metadata/wohns_pop_labels.csv")
colors_df = pd.read_csv("../data/metadata/wohns_pop_colors.csv")
Plotter = PlotEmbeddings(embeddings_df, labels_df, colors_df, generation_time=None)
Plotter.init_plot(time_scale="log")
Plotter.add_sample_trajectories(line_width=2, alpha=0.5)
pio.write_html(Plotter.fig, file="test.html", auto_open=True)

# %%
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
        self.generation_time = generation_time
        self.embeddings_df = embeddings_df
        self.fig = None
        

    def init_plot(self, time_scale: str = "log"):
        self.fig = go.Figure()
        
        # Setting axis styles
        axis_style = dict(
            showline=True,
            showgrid=True,
            linecolor="black",
            linewidth=1,
            ticks="outside",
            tickcolor="black",
            gridcolor="rgba(0,0,0,0)",
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

        # Setting axis labels
        x_label = "X1"
        y_label = "X2"
        z_label = "Time (generations ago)" if self.generation_time is None else "Time (years ago)"

        # Setting gridlines and ticks
        # xy_grid = np.linspace(common_min, common_max, num_xy_ticks+2)[1:-1]
        min_time = self.embeddings_df["time"].min()
        min_time = 1 if min_time <= 0 else min_time
        max_time = self.embeddings_df["time"].max()
        z_min_decade = int(np.floor(np.log10(min_time)))
        z_max_decade = int(np.ceil(np.log10(max_time)))
        z_ticks = 10 ** np.arange(z_min_decade, z_max_decade + 1)

        self.fig.update_layout(
            scene=dict(
                xaxis=dict(title=x_label, range=x_range, **axis_style),
                yaxis=dict(title=y_label, range=y_range, **axis_style),
                zaxis=dict(title=z_label, tickvals = z_ticks, ticktext = [str(int(z)) for z in z_ticks], **axis_style, type=time_scale)
            )
        )


    def add_sample_trajectories(self,line_width: int = 2,alpha: float = 1.0):
        samples = self.embeddings_df["sample_id"].unique()
        for sample in samples:
            sample_data = self.embeddings_df[self.embeddings_df["sample_id"] == sample]
            population = sample_data["population"].values[0]
            color = sample_data["color"].values[0]
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
                    showlegend=False
                )
            )
#%%

    # def add_population_medians(self):

    # def add_population_convex_hulls(self:
    # def add_iso_surfaces(self):

    # def basic_plot(self,min_time,max_time, log_time: bool = True):
    #     fig = px.line_3d(self.embeddings_df, x="X1", y="X2", z="time", color='population', line_group='sample_id', color_discrete_map=self.colors_dict,log_z=log_time)
    #     # Set xlim and ylim to be the same
    #     x_min, x_max = self.embeddings_df["X1"].min(), self.embeddings_df["X1"].max()
    #     y_min, y_max = self.embeddings_df["X2"].min(), self.embeddings_df["X2"].max()
    #     lim_min = min(x_min, y_min)
    #     lim_max = max(x_max, y_max)
    #     fig.update_layout(scene=dict(
    #         xaxis=dict(range=[lim_min, lim_max]),
    #         yaxis=dict(range=[lim_min, lim_max])
    #     ))
    #     fig.update_layout(showlegend=True)
    #     return fig

    # def dynamic_plot(self,log_time: bool = True):
    #     fig = px.line_3d(self.embeddings_df, x="X1", y="X2", z="time", color='population', line_group='sample_id', color_discrete_map=self.colors_dict,log_z=log_time)
    #     # Set xlim and ylim to be the same
    #     xy_min = min(self.embeddings_df["X1"].min(), self.embeddings_df["X2"].min())
    #     xy_max = max(self.embeddings_df["X1"].max(), self.embeddings_df["X2"].max())
    #     lim_min = min(xy_min, xy_min)
    #     lim_max = max(xy_max, xy_max)
    #     # Add 10% padding for margins
    #     padding = 0.1 * (xy_max - xy_min)
    #     xy_range = [xy_min - padding, xy_max + padding]
    #     fig.update_layout(scene=dict(
    #         xaxis=dict(range=[lim_min, lim_max]),
    #         yaxis=dict(range=[lim_min, lim_max])
    #     ))
    #     fig.update_layout(showlegend=True)
    #     return fig
    
    
    # def static_3d_plot(self, colors_dict, x_view=2,y_view=2, z_view=1):
    #     fig = go.Figure()
    #     fig.update_layout(
    #         scene=dict(
    #             xaxis_title='X1',
    #             yaxis_title='X2',
    #             zaxis_title='Time (generations ago)' if self.generation_time is None else 'Time (years ago)',
    #             zaxis_type='log' if self.log_time else 'linear'
    #         )
    #     )

    #     fig.add_trace(go.Scatter3d(
    #         x=self.embeddings_df["X1"],
    #         y=self.embeddings_df["X2"],
    #         z=self.embeddings_df["time"],
    #         mode='lines',
    #         line=dict(width=0.5, color=colors_dict[self.embeddings_df["population"].values[0]]),
    #         opacity= 1
    #         ))

    #     # Get combined min/max from X1 and X2 columns
    #     xy_min = min(self.embeddings_df["X1"].min(), self.embeddings_df["X2"].min())
    #     xy_max = max(self.embeddings_df["X1"].max(), self.embeddings_df["X2"].max())

    #     # Add 10% padding for margins
    #     padding = 0.1 * (xy_max - xy_min)
    #     xy_range = [xy_min - padding, xy_max + padding]

    #     fig.update_layout(showlegend=False,
    #         margin=dict(l=0, r=0, t=0, b=0),
    #         scene=dict(
    #             xaxis_title='X1',
    #             yaxis_title='X2',
    #             zaxis_title='Time (generations ago)',
    #             zaxis_type='log'
    #         )
    #     )

    #     fig.update_layout(
    #         scene=dict(
    #             xaxis=dict(range=xy_range),  # Same range for X
    #             yaxis=dict(range=xy_range),  # Same range for Y
    #             aspectmode='manual',  # Prevents auto-scaling
    #             aspectratio=dict(x=1, y=1, z=0.7),  # X/Y equal, Z scaled down
    #         )
    #     )

    #     fig.update_layout(
    #         width=600,
    #         height=600,
    #         scene_camera=dict(
    #             eye=dict(x=x_view, y=y_view, z=z_view),  # Diagonal view
    #         )
    #     )

    #     #fig.write_image(os.path.join(output,"static_3d_plot.png"), scale=3)
        
    #     return fig


# # %%
# def reshape_embeddings(df: pd.DataFrame) -> pd.DataFrame:
#     d1 = (df.query("component==1").drop(columns=['component','time_inx']).rename(columns={'embedding':'dimension 1'}))
#     d2 = (df.query("component==2").drop(columns=['component','time_inx']).rename(columns={'embedding':'dimension 2'}))
#     return pd.merge(d1,d2,on=['sample_id','population','time','sample_time'])

# def slice_convex_hull(points: np.ndarray) -> np.ndarray:
#     if len(points)<3: return points
#     center=points.mean(axis=0)
#     d=np.linalg.norm(points-center,axis=1)
#     keep=points[d<=np.percentile(d, CENTRAL_PERCENTILE*100)]
#     if len(keep)<3: return keep
#     hull=ConvexHull(keep)
#     return keep[hull.vertices]

# def rasterise_polygon(poly: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
#     if len(poly)<3: return np.zeros((len(ys),len(xs)),dtype=bool)
#     path=MplPath(poly)
#     xv,yv=np.meshgrid(xs,ys)
#     pts=np.vstack([xv.ravel(),yv.ravel()]).T
#     return path.contains_points(pts).reshape(len(ys),len(xs))

# def build_volume(df: pd.DataFrame,pop:str,grid_res:int,log_time:bool) -> Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
#     sub=df[df.population==pop]
#     xs_all,ys_all=sub['dimension 1'],sub['dimension 2']
#     pad_x=0.05*(xs_all.max()-xs_all.min())
#     pad_y=0.05*(ys_all.max()-ys_all.min())
#     xs=np.linspace(xs_all.min()-pad_x,xs_all.max()+pad_x,grid_res)
#     ys=np.linspace(ys_all.min()-pad_y,ys_all.max()+pad_y,grid_res)
#     times=np.sort(sub.time.unique())
#     ts=np.log(times) if log_time else times
#     vol=np.zeros((len(ts),grid_res,grid_res),dtype=bool)
#     for i,t in enumerate(times): vol[i]=rasterise_polygon(slice_convex_hull(sub[sub.time==t][['dimension 1','dimension 2']].values),xs,ys)
#     return vol,xs,ys,ts

# def surface_from_volume(vol:np.ndarray,xs:np.ndarray,ys:np.ndarray,ts:np.ndarray,sigma:float) -> Optional[Tuple[np.ndarray,np.ndarray]]:
#     blurred=gaussian_filter(vol.astype(float),sigma=sigma)
#     lo,hi=blurred.min(),blurred.max()
#     if hi<=lo: logging.warning("Empty volume; skipping iso-surface (%.3f–%.3f)",lo,hi); return None
#     level=0.5
#     if not(lo<level<hi): level=lo+0.5*(hi-lo); logging.warning("Adjusted iso level → %.3f (%.3f–%.3f)",level,lo,hi)
#     try: verts,faces,_,_=marching_cubes(blurred,level=level)
#     except RuntimeError as e: logging.warning("No surface found: %s; skipping",e); return None
#     nx,ny,nz=len(xs)-1,len(ys)-1,len(ts)-1
#     vx=xs[0]+verts[:,2]/nx*(xs[-1]-xs[0]); vy=ys[0]+verts[:,1]/ny*(ys[-1]-ys[0]); vz=ts[0]+verts[:,0]/nz*(ts[-1]-ts[0])
#     return np.stack([vx,vy,vz],axis=1),faces

# def median_df(df:pd.DataFrame,log_time:bool)->pd.DataFrame:
#     med=df.groupby('time',as_index=False)[['dimension 1','dimension 2']].median().sort_values('time')
#     med['time_plot']=np.log(med.time) if log_time else med.time
#     if MEDIAN_SMOOTH_WINDOW>1:
#         med['dimension 1']=med['dimension 1'].rolling(MEDIAN_SMOOTH_WINDOW,center=True,min_periods=1).mean()
#         med['dimension 2']=med['dimension 2'].rolling(MEDIAN_SMOOTH_WINDOW,center=True,min_periods=1).mean()
#     return med

# def plot_population(df:pd.DataFrame,pop:str,vol_data,fig:go.Figure,colour:str,mode:str,log_time:bool,show_traj:bool,show_median:bool,show_ribs:bool,show_surface:bool) -> None:
#     vol,xs,ys,ts=vol_data
#     sub=df[df.population==pop].copy()
#     sub['time_plot']=np.log(sub.time) if log_time else sub.time
#     if show_traj:
#         fig.add_traces(
#             px.line_3d(sub,x='dimension 1',y='dimension 2',z='time_plot',line_group='sample_id',hover_name='population')
#               .update_traces(line=dict(color=colour),opacity=INDIVIDUAL_LINE_ALPHA,
#                              hovertemplate='Population: %{hovertext}<br>Sample ID: %{customdata[0]}<br>Time: %{z:.2f}<extra></extra>',
#                              customdata=sub[['sample_id']].values).data)
#     if show_median:
#         med=median_df(sub,log_time)
#         fig.add_trace(go.Scatter3d(x=med['dimension 1'],y=med['dimension 2'],z=med['time_plot'],mode='lines',
#                                    line=dict(color=colour,width=MEDIAN_LINE_WIDTH),showlegend=False,legendgroup=pop,
#                                    hovertemplate='Population: '+pop+'<br>Time: %{z:.2f}<extra></extra>'))
#         if not med.empty:
#             z_base=ts[0]+0.05*(ts[-1]-ts[0])
#             fig.add_trace(go.Scatter3d(x=[med['dimension 1'].iloc[0]],y=[med['dimension 2'].iloc[0]],
#                                        z=[z_base],mode='text',text=[pop],textposition='bottom center',
#                                        textfont=dict(size=10,color=colour),showlegend=False,hoverinfo='none'))
#     if show_ribs:
#         for i,t in enumerate(ts):
#             mask=vol[i];
#             if mask.sum()<3: continue
#             rows,cols=np.nonzero(mask);coords=np.column_stack([xs[cols],ys[rows]]);
#             hull=slice_convex_hull(coords)
#             if len(hull)<3: continue
#             fig.add_trace(go.Scatter3d(x=hull[:,0],y=hull[:,1],z=np.full(len(hull),t),mode='lines',
#                                        line=dict(color=colour,width=1),showlegend=False,legendgroup=pop))
#     if show_surface and mode!='skeleton':
#         surf=surface_from_volume(vol,xs,ys,ts,GAUSS_SIGMA)
#         if surf is not None:
#             verts,faces=surf;i,j,k=faces.T
#             fig.add_trace(go.Mesh3d(x=verts[:,0],y=verts[:,1],z=verts[:,2],i=i,j=j,k=k,
#                                      color=colour,opacity=TUBE_ALPHA,name=pop,legendgroup=pop,showlegend=True))

# def build_plot(df:pd.DataFrame,palette:Dict[str,str],mode:str,log_time:bool,show_traj:bool,show_median:bool,show_ribs:bool,show_surface:bool)->go.Figure:
#     fig=go.Figure()
#     for pop in sorted(df.population.unique()):
#         colour=palette.get(pop,'#888888')
#         vol_data=build_volume(df,pop,GRID_RES_XY,log_time)
#         plot_population(df,pop,vol_data,fig,colour,mode,log_time,show_traj,show_median,show_ribs,show_surface)
#     fig.update_layout(scene=dict(xaxis_title='X1',yaxis_title='X2',
#                                  zaxis=dict(title='ln(time)' if log_time else 'time',
#                                             tickvals=[np.log(n) for n in [10,20,50,100,200,500,1000,2000,5000,10000,20000]],
#                                             ticktext=['10','20','50','100','200','500','1000','2000','5000','10000','20000'])),
#                       showlegend=True,margin=dict(l=0,r=0,t=0,b=0))
#     return fig
# %%
