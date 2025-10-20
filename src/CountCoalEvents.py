#%%
import numpy as np
import pandas as pd
import tskit
import os
import glob
from numba import njit
from tqdm import tqdm
from pathos.multiprocessing import ProcessingPool as Pool

class CountCoalEvents:
    def __init__(self,
                 output_dir: str,
                 ts_dir: str = None,
                 demes_file: str = None,
                 time_points: list = None,
                 window_sizes: list = None,
                 sample_ids: list = None,
                 admix_samples: list = None,
                 K: int = None,
                 num_posterior_samples: int = None,
                 thinning_fraction: float = 0.1):
        """Base class for counting coalescent events in tree sequences.

        Args:
            output_dir (str): Directory to save counts dataframes.
            ts_dir (str): A directory containing .trees files for a list of tree sequences.
            demes_file (str, optional): A demes file specifying the demographic model. Only provided if simulating independent trees. Defaults to None.
            time_points (list, optional): A list of time points for counting the coalescent events. Defaults to None.
            window_sizes (list, optional): A list of window sizes for each time point. Defaults to None.
            sample_ids (list, optional): A list of sample ids to compute counts for in the tree sequence. Defaults to None.
            admix_samples (list, optional): A list of samples that are treated as admixed. Defaults to None.
            K (int, optional): Number of ancestries, if admix_samples is provided. Defaults to None.
            num_posterior_samples (int, optional): Number of posterior ARG samples if ARGs where samples rather than providing a single point estimate. Defaults to None.
                If provided, contigs in the ts_dir should be appropriately named by adding a suffix _0, _1, ..., _num_posterior_samples-1 to each contig name.
            thinning_fraction (float, optional): Thinning fraction for each contig, where every pair of samples is chosen on average every thinning_fraction/n_samples number of trees. Defaults to 0.1.
        """

        # basic input validation
        if not os.path.isdir(ts_dir):
            raise ValueError(f"ts_dir is not a valid directory: {ts_dir}")
        
        if ts_dir is None:
            assert demes_file is not None and time_points is not None and window_sizes is not None, "if demes_file is provided, time_points, window_sizes, and sample_ids must also be provided"

        if ts_dir is not None:
            # locating tree sequence files and counting contigs and posterior samples
            ts_files = glob.glob(os.path.join(ts_dir, "*.trees"))
            if len(ts_files) == 0:
                raise ValueError(f"No .trees files found in directory: {ts_dir}")
            if num_posterior_samples is None:
                num_contigs = len(ts_files)
            else:
                num_contigs = len(ts_files) // num_posterior_samples
                if len(ts_files) % num_posterior_samples != 0:
                    raise ValueError(f"Number of .trees files ({len(ts_files)}) is not divisible by num_posterior_samples ({num_posterior_samples})")

            # loading tskit tree sequence to get sample information and default time points if not defined
            ts = tskit.load(ts_files[0])
            if sample_ids is None:
                sample_ids = ts.samples()
                self.num_samples = len(sample_ids)
                self.subsample = False # using all samples
            else:
                sample_ids = np.array(sample_ids)
                self.num_samples = len(sample_ids)
                self.subsample = True # using a subset of samples
            if time_points is None:
                time_points = np.logspace(1, np.log10(np.quantile(ts.nodes_time,0.9)), 100, base=10)
            if window_sizes is None:
                window_sizes = np.array([np.mean(ts.nodes_time) /100] * len(time_points))
            elif isinstance(window_sizes,int):
                window_sizes = np.array([window_sizes] * len(time_points))
            elif len(window_sizes) != len(time_points):
                raise ValueError("window_sizes must be of the same length as time_points")
            window_sizes = np.array(window_sizes)
        
        # handling admixed samples if provided
            self.admix = False
            if admix_samples is not None:
                self.admix = True
                assert K is not None, "K (num ancestries) must be provided if admix_samples is provided"
                self.admix_samples = np.array(admix_samples)
                if not np.all(np.isin(admix_samples, sample_ids)):
                    raise ValueError("Some admix_samples are not in sample_ids")
                self.non_admix_samples = np.setdiff1d(sample_ids, admix_samples)

                self.num_admix = len(self.admix_samples)
                self.num_non_admix = len(self.non_admix_samples)
                self.num_idxs = self.num_non_admix + self.num_admix * K
                types = [0] * self.num_non_admix
                for k in range(K):
                    types += [k + 1] * self.num_admix
                sample_id_to_index = {}
                index_to_sample_id = {}
                for idx, sample_id in enumerate(self.non_admix_samples):
                    sample_id_to_index[sample_id] = idx
                    index_to_sample_id[idx] = sample_id
                for idx, sample_id in enumerate(self.admix_samples):
                    indices = len(self.non_admix_samples) + np.arange(0, K * self.num_admix, self.num_admix) + idx
                    sample_id_to_index[sample_id] = indices
                    for ind in indices:
                        index_to_sample_id[ind] = sample_id
                self.sample_id_to_index = sample_id_to_index

                # creating a DataFrame to map indices to sample IDs, individual IDs, and types
                # The counts matrix will be arranged according to this DataFrame
                id_df = pd.DataFrame(list(index_to_sample_id.items()), columns=["idx", "sample_id"])
                id_df["individual_id"] = id_df["sample_id"] // 2
                id_df["type"] = types
                id_df.sort_values(by="idx", inplace=True)
                id_df.reset_index(drop=True, inplace=True)
                self.id_df = id_df
            else:
                index_to_sample_id = {idx: sample_id for idx, sample_id in enumerate(sample_ids)}
                id_df = pd.DataFrame(list(index_to_sample_id.items()), columns=["idx", "sample_id"])
                id_df["individual_id"] = id_df["sample_id"] // 2
                id_df["type"] = 0
                id_df.sort_values(by="idx", inplace=True)
                id_df.reset_index(drop=True, inplace=True)
                self.id_df = id_df
                self.num_idxs = self.num_samples
        
        id_df.to_parquet(os.path.join(output_dir, "idx_to_sample.parquet"), index=False)
        self.output_dir = output_dir
        self.demes_file = demes_file
        self.ts_files = ts_files
        self.num_contigs = num_contigs
        self.num_posterior_samples = num_posterior_samples
        self.ts_dir = ts_dir
        self.num_time_points = len(time_points)
        self.time_points = time_points
        self.window_sizes = window_sizes
        self.sample_ids = sample_ids
        self.admix_samples = admix_samples
        self.K = K
        self.thinning_fraction = thinning_fraction
        self.pairs = np.column_stack(np.triu_indices(self.num_samples, k=1)) 
        self.num_pairs = self.pairs.shape[0]
        self.processed_contigs = False

    def count_coal_events_contig(self, trees, LAI_data=None):
        """A function to count coalescent events over time in a single contig.

        Args:
            trees (iterable): an iterator over trees in a tree sequence.
            LAI_data (pd.DataFrame, optional): A pandas dataframe with formatted LAI results. Defaults to None.

        Returns:
            np.ndarray: A tuple of two numpy arrays (N, L) representing coalescent event counts and uncoalesced loci counts.
        """
        if self.admix:
            assert LAI_data is not None, "LAI_data must be provided if admix_samples are used"
        # Adapt to account for admix samples as well
        if self.time_points[0] !=0:
            time_points = np.concatenate(([0], self.time_points))
            window_sizes = np.concatenate(([self.window_sizes[0]],self.window_sizes))
            remove_later = True
        else:
            time_points = self.time_points
            window_sizes = self.window_sizes
            remove_later = False
        
        times_upper = time_points + window_sizes

        N = np.zeros((self.num_idxs, self.num_idxs, len(time_points)), dtype=np.uint16) # matrix of number of coalescent events within sliding window 
        C = np.zeros((self.num_idxs, self.num_idxs, len(time_points)), dtype=np.uint16) # matrix specifying if i and j have coalesesced by t
        L0 = np.zeros((self.num_idxs, self.num_idxs), dtype=np.uint16) # matrix counting number of loci where pairs i and j were chosen

        for tree in trees:
            mask = np.random.random(size=self.num_pairs) < self.thinning_fraction/self.num_samples
            active_pairs = self.pairs[mask]
            tmrca_active_pairs = np.array([tree.tmrca(ip, jp) for ip, jp in active_pairs])
            ## Handling admixed samples case
            if self.admix:
                active_samples = np.unique(active_pairs)
                active_admix_samples = np.intersect1d(active_samples, self.admix_samples)
                # Finding local ancestry for active admixed samples
                if len(active_admix_samples) > 0:
                    local_ancestry = find_local_ancestry(active_admix_samples, tree.interval[0], LAI_data)
                # matching active pairs to their respective indices in the counts matrices
                for pair_idx,pair_id in enumerate(active_pairs):
                    sample_i = pair_id[0]
                    sample_j = pair_id[1]
                    active_pairs[pair_idx,0] = self.sample_id_to_index[sample_i] if sample_i in self.non_admix_samples else self.sample_id_to_index[sample_i][local_ancestry[sample_i]]
                    active_pairs[pair_idx,1] = self.sample_id_to_index[sample_j] if sample_j in self.non_admix_samples else self.sample_id_to_index[sample_j][local_ancestry[sample_j]]
            # Remove all rows where there is any nan value in active_pairs
            active_pairs = active_pairs[~np.isnan(active_pairs).any(axis=1)]
                    
            for pair,tmrca in zip(active_pairs,tmrca_active_pairs):
                L0[pair[0],pair[1]] += 1
                update_matrices(N,C, pair,tmrca, time_points, times_upper)

        L = L0[:,:,np.newaxis]-np.cumsum(C, axis=2)  # Number of uncoalesced loci
        if remove_later:
            N = N[:, :, 1:]  # Remove the first time point if it was added
            L = L[:, :, 1:]

        return N,L

    def count_coal_events_total(self, parallel: bool = True):
        """A method to count coalescent events over time for all contigs and save results as parquet files.
        """
        def process_ts_file(ts_file):
            if self.subsample:
                ts = tskit.load(ts_file).simplify(self.sample_ids)
            else:
                ts = tskit.load(ts_file)
            trees = ts.trees()
            N, L = self.count_coal_events_contig(trees)
            df = counts_to_df(N, L, self.time_points)
            contig_name = os.path.basename(ts_file).replace(".trees", "")
            out_path = os.path.join(self.output_dir, f"{contig_name}_counts.parquet")
            df.to_parquet(out_path)
        
        if parallel:
            with Pool(os.cpu_count()) as pool:
                for ts_file in tqdm(pool.imap(process_ts_file, self.ts_files), total=len(self.ts_files)):
                    pass
        else:
            for ts_file in tqdm(self.ts_files):
                _ = process_ts_file(ts_file)
        self.processed_contigs = True
        
    def aggregate_counts(self):
        """A method that aggregates counts across contigs and posterior samples if provided.

        Returns:
            pd.DataFrame: Aggregated counts across contigs and posterior samples.
        """
        assert self.processed_contigs, "You must run count_coal_events_total() before aggregating counts."
        count_files = glob.glob(os.path.join(self.output_dir, "*_counts.parquet"))
        # loading base data from a single counts file
        sample_df = pd.read_parquet(count_files[0])
        sample_1_idx = sample_df["sample_1_idx"].values
        sample_2_idx = sample_df["sample_2_idx"].values
        time_idx = sample_df["time_idx"].values
        time = sample_df["time"].values
        total_counts_df = pd.DataFrame({
            "sample_1_idx": sample_1_idx,
            "sample_2_idx": sample_2_idx,
            "time_idx": time_idx,
            "time": time})

        if self.num_posterior_samples is None:
            num_uncoalesced = np.zeros_like(sample_df["num_uncoalesced"].values)
            num_coal_events = np.zeros_like(sample_df["num_coal_events"].values)
            for count_file in count_files:
                df = pd.read_parquet(count_file)
                num_uncoalesced += df["num_uncoalesced"].values
                num_coal_events += df["num_coal_events"].values
            total_counts_df["num_uncoalesced"] = num_uncoalesced
            total_counts_df["num_coal_events"] = num_coal_events
            total_counts_df["coal_rate"] = num_coal_events / (num_uncoalesced*self.window_sizes[time_idx])

        else:
            coal_rates = np.zeros((len(sample_df["num_coal_events"].values),self.num_posterior_samples), dtype=float)
            for i in range(self.num_posterior_samples):
                # Sum counts across contigs for posterior sample i
                count_files_i = [f for f in count_files if f.endswith(f"_{i}_counts.parquet")]
                num_uncoalesced = np.zeros_like(sample_df["num_uncoalesced"].values)
                num_coal_events = np.zeros_like(sample_df["num_coal_events"].values)
                for count_file in count_files_i:
                    df = pd.read_parquet(count_file)
                    num_uncoalesced += df["num_uncoalesced"].values
                    num_coal_events += df["num_coal_events"].values
                # Calculating coal rate for posterior sample i and adding to running total
                coal_rates[:, i] = num_coal_events / (num_uncoalesced*self.window_sizes[time_idx])
            # Calculating average coal rate across posterior samples
            total_counts_df["coal_rate"] = np.nanmedian(coal_rates, axis=1)

        total_counts_df.to_parquet(os.path.join(self.output_dir, "total_counts.parquet"))
        return total_counts_df

    def sim_indep_trees(self, n=50, L0 = 1e4, random_seed = 42):
        """A function to simulate independent trees under a demes model using msprime.

        Args:
            n (int, optional): Number of haploid samples per population. Defaults to 50.
            L0 (_type_, optional): Total number of independent trees to simulate. Defaults to 1e4.
            random_seed (int, optional): Random seed for reproducibility. Defaults to 42.

        Returns:
            _type_: _description_
        """
        import msprime
        import demes
        graph = demes.load(self.demes_file)
        demography = msprime.Demography.from_demes(graph)
        K=0
        samples = {}
        population_samples_dict = {}
        extant_populations = []
        for pop in demography.populations:
            if pop.default_sampling_time == 0:
                samples[pop.name] = n
                extant_populations.append(pop.name)
                population_samples_dict[pop.name] = np.arange(n*K,n*(K+1))
                K+=1

        n_samples = n * K
        ts_list = msprime.sim_ancestry(
            samples = samples,
            demography = demography,
            sequence_length=1,
            ploidy = 1,
            num_replicates = L0*n_samples/self.thinning_fraction,
            random_seed = random_seed
            )
        
        self.num_samples = n_samples
        self.num_idx = n_samples
        self.sample_ids = np.arange(n_samples)
        self.subsample = False
        trees = (ts.first() for ts in ts_list)
        return trees
    


@njit
def update_matrices(N:np.ndarray, C:np.ndarray, pair:tuple, tmrca:float, times:np.ndarray, times_upper:np.ndarray):
    """Update the count matrices for coalescent events and coalesced lineages.

    Args:
        N (np.ndarray): matrix of number of coalescent events within sliding window.
        C (np.ndarray): matrix of number of coalesced lineages within sliding window.
        pair (tuple): pair of samples to be updated
        tmrca (float): tmrca of that pair
        times (np.ndarray): time points at which to count coalescent events
        times_upper (np.ndarray): upper bounds of time intervals
    """
    i, j = pair
    for k in range(len(times)):
        if tmrca > times[k]: # Has not coalesced yet
            if tmrca <= times_upper[k]: # Within the sliding window
                N[i, j, k] += 1
        else: # Has coalesced
            C[i, j, k] += 1
            break

def find_local_ancestry(active_admix_samples,start_coordinate,LAI_data):
    local_ancestry = {sample_id:np.nan for sample_id in active_admix_samples}
    LAI_data_subset = LAI_data[(LAI_data.spos < start_coordinate) & (LAI_data.epos >= start_coordinate)]
    if LAI_data_subset.empty:
        LAI_data_subset = LAI_data[(LAI_data.spos == min(LAI_data.spos))]
    LAI_data_subset = LAI_data_subset[LAI_data_subset.sample_id.isin(active_admix_samples)]
    for _,row in LAI_data_subset.iterrows():
        try:
            local_ancestry[row.sample_id] = int(row.ancestry)
        except:
            continue
    return local_ancestry

def counts_to_df(N,L,times):
    """Convert count matrices to a pandas DataFrame."""
    n_total = N.shape[0]
    triu_indices = np.triu_indices(n_total, k=1)
    sample_1_idx = np.repeat(triu_indices[0], len(times))
    sample_2_idx = np.repeat(triu_indices[1], len(times))
    time_idx = np.tile(np.arange(len(times)), len(sample_1_idx) // len(times))
    time = np.tile(times, len(sample_1_idx) // len(times))
    num_uncoalesced = L[triu_indices[0], triu_indices[1], :].flatten()
    num_coal_events = N[triu_indices[0], triu_indices[1], :].flatten()

    df = pd.DataFrame({
        "sample_1_idx": sample_1_idx,
        "sample_2_idx": sample_2_idx,
        "time_idx": time_idx,
        "time": time,
        "num_uncoalesced": num_uncoalesced,
        "num_coal_events": num_coal_events,
    })

    return df
# %%
