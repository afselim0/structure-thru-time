#!/usr/bin/env python3
import argparse
import sys
import pandas as pd
sys.path.append("../../src")
sys.path.append("../src")
from PlotEmbeddings import PlotEmbeddings

def main():
    parser = argparse.ArgumentParser(description="Plot embeddings")
    parser.add_argument("--input", required=True, help="Path to embeddings parquet file")
    parser.add_argument("--colors", required=True, help="Path to population colors parquet file")
    parser.add_argument("--labels", required=True, help="Path to population labels parquet file")
    parser.add_argument("--output_path", required=True, help="Output directory for plots")
    parser.add_argument("--output_name", required=True, help="Output filename for plot")
    parser.add_argument("--min_time", default=0, type=float,help="Minimum time for plotting")
    parser.add_argument("--generation_time", default=None, type=float,help="Generation time in years")

    print("Reading files...")
    args = parser.parse_args()
    embeddings_df = pd.read_parquet(args.input)
    colors_df = pd.read_csv(args.colors)
    labels_df = pd.read_csv(args.labels)
    output_path = args.output_path
    output_name = args.output_name
    min_time = args.min_time
    generation_time = args.generation_time
    
    # subsetting labels_df and colors_df to active samples only
    active_samples = embeddings_df.sample_id.unique()
    labels_df = labels_df[labels_df['sample_id'].isin(active_samples)]
    populations = labels_df['population'].unique()
    colors_df = colors_df[colors_df['population'].isin(populations)]

    print("Building plots...")

    embeddings_df = embeddings_df[embeddings_df['time'] >= min_time]
    plotter = PlotEmbeddings(embeddings_df,labels_df,colors_df,generation_time=generation_time)
    plotter.init_plot(time_scale="log")
    plotter.add_sample_trajectories()
    plotter.add_population_medians()
    plotter.add_legend()

    plotter.fig.write_html(f"{output_path}/{output_name}")
    plotter.save_static(filename=f"{output_path}/{output_name.removesuffix('.html')}_v1.png", format="png", width=600, height=600, scale=2, eye=dict(x=1.8, y=1.8, z=0.3), up=dict(x=0, y=0, z=1))
    plotter.save_static(filename=f"{output_path}/{output_name.removesuffix('.html')}_v2.png", format="png", width=600, height=600, scale=2, eye=dict(x=-0.75, y=-1.5, z=0.3), up=dict(x=0, y=0, z=1))
    #embeddings_dyn_fig.show()

    # embeddings_dyn_fig.update_traces(opacity=1,line=dict(width=1))
    # embeddings_dyn_fig.update_layout(
    #     width=600,
    #     height=600,
    #     paper_bgcolor="white",
    #     scene=dict(
    #         camera=dict(
    #             eye=dict(x=1.8, y=1.8, z=0.3),
    #             up=dict(x=0, y=0, z=1),
    #         ),
    #         aspectmode="data",
    #         zaxis_title='Time (generations ago)',
            
    #         xaxis=dict(nticks=5, showbackground=True, backgroundcolor="whitesmoke", 
    #                     showgrid=True, gridcolor="lightgray",
    #                     showline=True,
    #                     linecolor="black",
    #                     linewidth=2,
    #                     zeroline=True),
    #         yaxis=dict(nticks=5, showbackground=True, backgroundcolor="whitesmoke", 
    #                     showgrid=True, gridcolor="lightgray",
    #                     showline=True,
    #                     linecolor="black",
    #                     linewidth=2,
    #                     zeroline=True),
    #         zaxis=dict(nticks=5, showbackground=True, backgroundcolor="whitesmoke", 
    #                     showgrid=True, gridcolor="white",
    #                     showline=True,
    #                     linecolor="black",
    #                     linewidth=2,zeroline=True),
    #     )
    # )
    # static_output_filename = f"{output_path}/{output_name}".removesuffix(".html")
    # embeddings_dyn_fig.write_image(f"{static_output_filename}.pdf",scale=2)
     
     
    # embeddings_dyn_fig.update_layout(
    #     scene=dict(
    #         camera=dict(
    #             eye=dict(x=-0.75, y=-1.5, z=0.3),
    #             up=dict(x=0, y=0, z=1),
    #         )
    #     )
    # )
    # embeddings_dyn_fig.write_image(f"{static_output_filename}_v2.pdf",scale=2)
    
    # embeddings_static_fig=plotter.static_3d_plot()
    # embeddings_static_fig.write_image(f"{static_output_filename}_v3.pdf",scale=2)
    
if __name__ == "__main__":
    main()
