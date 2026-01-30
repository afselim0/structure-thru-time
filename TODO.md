# outputs / workflow engineering
- Finalize performance evaluation

- Wohns et al and Chimp et al
    - PAUSED same large chrom, chimps and humans, singer relate  [Chimp takes too long to run]
        - choose a syntenic hom chrom
    - ONGOING Getting it to run and match the signal observed in previous 
    runs
      - Goal:  Focused on chr 21
      - 4-pop HPRC (Please post output files to project/jnovembre)
         - GO Double SINGER params and re-run chr 21 as needed [ Haoyang]
         - GO Relate - chr 21
         - DONE Relate trees - chr 20, 22
         - Run the "all" and "combs" rule [ Ahmed ]
      - 26-pop HPRC (4 per pop)  
         - DONE Relate trees - chr 20, 22   
         - GO Relate - Chr 21
            - Relate tree path: /project/jnovembre/hyhu/work/relate/chr21/chr21_tree
         - GO Double check SINGER params and re-run chr 21 as needed [ Haoyang ]
            - SINGER trees path: /project/jnovembre/hyhu/work/singer/chr21/chr21_tree
         - Once ready, Run the "all" rule 
      - Wohns (1 per pop) - Bonus 
      - Chimps      
          - DONE chr 21 - whole chromosome with n=10(?) per pop
          - DONE Nuria's Singer chr 21 
          - Try a run that subsets chr 21 more aggressively but max's out n [ Haoyang ]
             - Chop post-singer chr 21 into chunks and running stt with the goal of running larger sampler set [ Haoyang ]  
         - Run the "all" and the "combs" rule 

    - Revised set of preliminary figures 
    - AT For manuscript - template figure in the latex 
    - AT In the working directories - several figure outputs;     

    - Master snakefile for running both simulations and empirical results
    - Parent directories in /project/jnovembre to keep track of results
    - Text and colors in embedding files to capture residuals
    - Add procrusted and penalty
    - add max distance
    

- Adding option for Singer vs tsinfer/tsdate, and relate
    - add to simulation pipeline
    - For Wohns et al - output to vcf and reinfer with SINGER 

- Turn attention to the local admixture problem 

Minor:
- Elaborating the snakefile for manuscript figure generation: In Figures/snakefile
    - Rule to make the single pop fig as a template for other figures
- Set up syncronization so that overleaf has access to fig outputs directory(ies)
- One or several "clean"-type rules in the snakemake that remove intermediate files that take up space and are quickly regenerated
- Outputs to log files... 
- Static figures cannot be saved on cluster


To Discuss:
- 1/27 Some confusion with wohns_pop_labels and colors? 
- 1/26 Give thought to the natural boundary case observation of most recent window with no coalescence between any one.  That point will just be pushed away from all others.  What does that suggest about initialization / benefits of sharing information across windows etc. 
- 1/26 Next paper or a small last section - use embedding to re-estimate rates and show an improvement over the raw rates
- 1/26 Revisit Procrustes analysis per time step option
- 1/26 Show multiple solutions for same data (HT Amy G)
- 1/26 2-D plots wuold help (also for showing admixture) (HT Amy G)
- 1/26 What about coal rate variation across loci / additional source so variation? (HT Amy G) my idea - Could add a locus-specific parameter to distance function for background selection?  And an extra noise layer? 
- 1/26 Behavior of N0 reported by Ahmed suggest maybe we could revisit the use of N0 - maybe need two parameters - to effect two extremes differernetly (bottleneck - high lambda vs divergence low lambda)
- 1/26 Add outputs for diagnostics to embedding script - csv - # of iterative steps per time t; loss function value achieved per time t;   
- 1/26 I think we need plotting / animations for time slices through the embeddings - as complement for colorblind - use abbreviations per individual (e.g numbers or two letter)
- 1/26 We should do a forensic analysis of why the other chimp analysis is so different -- is it change in distance functino? Other tunables? 
- 1/26 What conidtions allowed us to the Wohns with 900 individuals and what specific changes are holding us back now?  
- 1/26 Need to profile memory usage and if we can get control of it 
- 1/26 Future direction: admixture is pernicious too in that it pulls source populations closer together than they should be; Can we downweight attraction to samples who themselves have highby cross locus variance in weights - so if sample has high variance - it's attracted to others, but others are not attracted to it; might work in an alternate distance formulation; Would be easier in a logistic link 
- 1/26 Fixed L derivation so that we have likelihoods given an assumed effective sample size of $L$
- 1/24 How to deal with issue that when lambda = 0 between any pair exact spacing will depend on BFGS tolerance - we should make this more explicit perhaps; e.g. lambda = 1/(N0+d_ij)-1/Nmax ; where Nmax is say N_0*10^5 for default 
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
- DONE In sim snakemake - Update True embedding outputs to mark choices of N0 and other specific relevant params 
- DONE Polish the rate inference - because of newly appreciated issues with recent times