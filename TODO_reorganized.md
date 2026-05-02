# General Tasks
- **Smoothing the Embeddings**
- **Performance Evaluation**
- **Finalize Empirical Analyses**

# Specific Tasks
- **Smoothing the Embeddings**
    - Implement a forward backward pass with no explicit smoother
    - track loss function 
    - Determine a standard level of smoothing given standard error of rate estimates
    - Add necessary parameters and changes to snakemake
        - A basic unsmoothed output
        - An output with the standard level of smoothing
        - Any additional outputs for additionally specified smoothing parameters

- **Performance Evaluation**
    - Look more into highlighting outliers
    - What's going on with split + tsinfer
    - Understanding the pulses in reconstruction error of complex 4 pop (does it go away by just increasing maximum window size?)
    - reconstructed rates vs true rates vs vanilla rate estimator from ARGs (simulations) [high]
        - d=2, d=10, d=n-1
        - True rates (from model) vs reconstructed rates or vanilla from our rate estimate
    - Special: Do admixed individuals pull their sources too much? [medium]
    - Add an option of multiple outputs from diff inits
    
- **Finalyzing Empirical Analyses**
    - Preparing datasets [medium]
        - Reinfer Wohns with tsinfer+tsdate
            - with and without archiacs
        - Reinfer Chimps with tsinfer+tsdate

# Minor Tasks
- Embeddings for individuals rather than haplotypes [high]
- Add outputs for diagnostics to embedding script - csv - # of iterative steps per time t; loss function value achieved per time t;   
- Elaborating the snakefile for manuscript figure generation: In Figures/snakefile
    - Rule to make the single pop fig as a template for other figures
- One or several "clean"-type rules in the snakemake that remove intermediate files that take up space and are quickly regenerated
- Outputs to log files... 

# Discussion Points
- 1/26 Next paper or a small last section - use embedding to re-estimate rates and show an improvement over the raw rates
- 1/26 Show multiple solutions for same data (HT Amy G)

- 1/26 Give thought to the natural boundary case observation of most recent window with no coalescence between any one.  That point will just be pushed away from all others.  What does that suggest about initialization / benefits of sharing information across windows etc. 

- 1/26 Need to profile memory usage and if we can get control of it - short term seems like lowering FP helps
- 1/31 Any analogy / relationshio of FP to stochastic gradient? 

- 1/26 Behavior of N0 reported by Ahmed suggest maybe we could revisit the use of N0 - maybe need two parameters - to effect two extremes differernetly (bottleneck - high lambda vs divergence low lambda)

- 1/26 We should do a forensic analysis of why the other chimp analysis is so different -- is it change in distance functino? Other tunables? 
- 1/26 What conidtions allowed us to the Wohns with 900 individuals and what specific changes are holding us back now?  
- 1/27 Some confusion with wohns_pop_labels and colors? 

- 1/26 What about coal rate variation across loci / additional source so variation? (HT Amy G) my idea - Could add a locus-specific parameter to distance function for background selection?  And an extra noise layer? 
- 1/26 Future direction: admixture is pernicious too in that it pulls source populations closer together than they should be; Can we downweight attraction to samples who themselves have highby cross locus variance in weights - so if sample has high variance - it's attracted to others, but others are not attracted to it; might work in an alternate distance formulation; Would be easier in a logistic link 
- 1/26 Fixed L derivation so that we have likelihoods given an assumed effective sample size of $L$
- 1/24 How to deal with issue that when lambda = 0 between any pair exact spacing will depend on BFGS tolerance - we should make this more explicit perhaps; e.g. lambda = 1/(N0+d_ij)-1/Nmax ; where Nmax is say N_0*10^5 for default 
- 1/18 Results are coming in squished on the 2nd dimension?  
structure_thru_time/results/simulations/figs_all/ooa_4_no_mig_embeddings_true.html
- 1/18 Where are the empirical outputs... let's be sure to keep example runs


## Archive
#### Done
- DONE Parent directories in /project/jnovembre to keep track of results

- DONE 2-D plots wuold help (also for showing admixture) (HT Amy G)
- DONE From discussion: 1/26 I think we need plotting / animations for time slices through the embeddings - as complement for colorblind - use abbreviations per individual (e.g numbers or two letter)
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
- DONE Revisit Procrustes analysis per time step option [ Ahmed ]
- DONE Penalized inference
- DONE add max distance
- DONE Adding option for Singer vs tsinfer/tsdate, and relate [ Ahmed ]
    - add to simulation pipeline
- DONE Master snakefile for running both simulations and empirical results
- DONE Finish SINGER inference on simulated data
- DONE Add the sample-based coordinate descent to the LatentSpaceEmbeddings.py script 
- Done Rerun chimps asap to identify what's gone awry since the 2025 set of results 
- DONE reconstructed rates vs input rates [high]
- DONE **Finalize choice of ARG inference method: Use tsinfer + new tsdate**
    - Add a wrapper .sh script that runs the simulations + all 3 inference methods + STT on all of them [medium]
    - In simulations, compare trees inferred using SINGER with those inferred using tsinfer [medium]
    - Compare rates inferred from SINGER trees vs tsinfer trees (recent history?) [medium]
    - Follow up with Yan about new tsinfer/tsdate + new Wohns data

### Archived discussion points
- Wohns et al and Chimp et al for UCLA talk 
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
