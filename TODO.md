# simulations

## outputs / workflow engineering

- Polish the rate inference - because of newly appreciated issues with recent times [Ahmed]

- Finalize performance evaluation

- Wohns et al and Chimp et al
    - PAUSED same chrom, chimps and humans, singer relate  [Chimp takes too long to run]
        - choose a syntenic hom chrom
    - ONGOING Getting it to run and match the signal observed in previous runs
      - Goal:  Focused on chr 21
      - 4-pop HPRC (Please post output files to project/jnovembre)
         - Run the "combs" rule [ Ahmed ]
      - 26-pop HPRC (4 per pop) 
         - Double check SINGER params and re-run as needed
         - Once ready,  re-run with latest scripts changes
      - Wohns (1 per pop) 
      - Chimps 
         - Try a run that subsets the genome more aggressively 
         - Chop post-singer chr 21 into chunks and running stt with the goal of running larger sampler set [ Haoyang ]  
         - Run the "combs" rule 
 

    - Revised set of preliminary figures 
    - AT For manuscript - template figure in the latex 
    - AT In the working directories - several figure outputs;     


- Adding option for Singer vs tsinfer/tsdate, and relate
    - add to simulation pipeline
    - For Wohns et al - output to vcf and reinfer with SINGER 

- Turn attention to the local admixture problem 

Minor:
- DONE In sim snakemake - Update True embedding outputs to mark choices of N0 and other specific relevant params 
- Elaborating the snakefile for manuscript figure generation: In Figures/snakefile
    - Rule to make the single pop fig as a template for other figures
- Set up syncronization so that overleaf has access to fig outputs directory(ies)
- One or several "clean"-type rules in the snakemake that remove intermediate files that take up space and are quickly regenerated
- Outputs to log files... 


To Discuss:
- 1/18 Results are coming in squished on the 2nd dimension?  
structure_thru_time/results/simulations/figs_all/ooa_4_no_mig_embeddings_true.html
- 1/18 Where are the empirical outputs... let's be sure to keep example runs

# Misc:
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
- DONE Performance evaluation script - 
  - How well does the embedding represent the input rate matrix?  
     - Through time across all lineages and on average over time
     - How well per lineage per moment in time (and perhaps use that as color in plot!)
- DONE the population combos script developed for chimps should be brought in as a general utility script
- DONE Getting outputs similar to sim structure so same util scripts can run on it 