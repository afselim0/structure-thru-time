# simulations

## outputs / workflow engineering
- Performance evaluation script - 
  - How well does the embedding represent the input rate matrix?  
     - Through time across all lineages and on average over time
     - How well per lineage per moment in time (and perhaps use that as color in plot!)

- Wohns et al and Chimp et al  
    - Getting it to run and match the signal observed in previous runs
      - Goal: 
      - 4-pop (Please post output files to project/jnovembre)
      - 1 per pop Wohns 
      - Chimps 
 
    - Revised set of preliminary figures 
    - For manuscript - template figure in the latex 
    - In the working directories - several figure outputs; 
    - One pop at a time - plus the joint
    - DONE Getting outputs similar to sim structure so same util scripts can run on it 

- Adding option for Singer vs tsinfer/tsdate
    - For Wohns et al - output to vcf and reinfer with SINGER 

- Turn attention to the local admixture problem 

Minor:
- In sim snakemake - Update True embedding outputs to mark choices of N0 and other specific relevant params 
- Elaborating the snakefile for manuscript figure generation: In Figures/snakefile
    - Rule to make the single pop fig as a template for other figures
- Set up syncronization so that overleaf has access to fig outputs directory(ies)
- One or several "clean"-type rules in the snakemake that remove intermediate files that take up space and are quickly regenerated
- Outputs to log files... 

To Discuss:
- 1/20 Quite puzzled by performance issue that the predicted rates from spatial positions are not scaled correctly -- scaling the loss function by a constant should not change the minimizing value right; is there just a bug somewhere?  
- 1/20 Check factors of Delta_t 
- 1/20 Line 52 of evaluate_performance.py - looks almost like it re-runs the embedding? 
- 1/19 Are the demes files for human examples stdpopsim all handy?  I think you exported them from stdpopsim at some point for doing the gallery of human examples? 
- 1/18 Results are coming in squished on the 2nd dimension?  
structure_thru_time/results/simulations/figs_all/ooa_4_no_mig_embeddings_true.html
- 1/18 the population combos script developed for chimps should be brought in as a general utility script
- 1/18 Where are the empirical outputs... let's be sure to keep example runs

Misc:


### Moot discussion points

- 1/18 JN : Didn't accomplish too much except to realize how fussy plotly static export is!  Static figures are improved somewhat; 2 camera outputs; and figure scraper is set up; see note above about need to step up and link in Luke's code  
- 1/18 Snakemake stalls oddly on split.yaml - not sure why - some kind of lock or sync issue
- 1/18 Realizing I don't know how to launch a job on cluster that will run in a particular conda environment - add source activate line to bash script that call snakemake
- 1/18 About an 1hr 20 min to do all deme examples from scratch with FP=1 on laptop!


### Done
- DONE Pulling in Luke's plotting code to be plotting script by default; Also there is some existing static export code in PlotEmbeddings that has some upsides but it has some quirks in its plotting.  
- DONE Scraper of all output figs into one directory for easy exploration of results
- DONE Use csv vs parquet for popcolors/poplabels so that manual editing of color schemes is feasible
- DONE Calculating rates is a seperate step after estimating the binned_pdf in snakemake