#%%
import numpy as np
import pandas as pd
import tskit
import os
from pathlib import Path
from tqdm import tqdm
from pathos.multiprocessing import ProcessingPool as Pool
import re
from scipy.signal import savgol_filter

class CountCoalEvents:
    def __init__(self,
                 ts_dir: str,
                 sample_ids: list = None,
                 num_posterior_samples: int = None,
                 max_time: float = None,
                 num_timepoints: int = 100,
                 fraction_pairs: float = 1.0):
        
        self.ts_dir = ts_dir
        self.num_timepoints = num_timepoints
        self.fraction_pairs = fraction_pairs
        print("preprocessing tree sequence files...")
        # basic input validation
        assert os.path.isdir(ts_dir), f"ts_dir is not a valid directory: {ts_dir}"

        # locating tree sequence files and counting contigs and posterior samples
        self.ts_files = [file.name for file in Path(ts_dir).glob("*.trees")]
        if len(self.ts_files) == 0:
            raise ValueError(f"No .trees files found in directory: {ts_dir}")
        self.ts_files.sort()
        if num_posterior_samples is None:
            self.num_posterior_samples = 1
            self.num_contigs = len(self.ts_files)
            self.contig_ids = [ts_file.rstrip('.trees') for ts_file in self.ts_files]
            self.contig_lengths = {contig_id:tskit.load(os.path.join(ts_dir, contig_id + ".trees")).sequence_length for contig_id in tqdm(self.contig_ids,total=self.num_contigs)}
        else:
            self.num_posterior_samples = num_posterior_samples
            self.num_contigs = len(self.ts_files) // num_posterior_samples
            self.contig_ids = np.unique([re.sub(r'_[0-9]+\.trees$', '', contig) for contig in self.ts_files]) ## Must have the format {ContigID_PosteriorSampleID.trees}
            if len(self.ts_files) % num_posterior_samples != 0:
                raise ValueError(f"Number of .trees files ({len(self.ts_files)}) is not divisible by num_posterior_samples ({num_posterior_samples})")
            self.contig_lengths = {contig_id:tskit.load(os.path.join(ts_dir, contig_id + "_0.trees")).sequence_length for contig_id in tqdm(self.contig_ids,total=self.num_contigs)}
        

        # saving sample data
        ts = tskit.load(os.path.join(ts_dir, self.ts_files[0]))
        self.sample_sets = [[sample_id] for sample_id in ts.samples()]
        if sample_ids is None:
            sample_ids = ts.samples()
            self.num_samples = len(sample_ids)
        else:
            sample_ids = np.array(sample_ids)
            sample_ids = np.sort(sample_ids)
            self.num_samples = len(sample_ids)

        if max_time is None:
            max_time = np.quantile(ts.nodes_time, 0.95)
        self.max_time = max_time

        # creating pairs of samples
        sample_pairs = []
        for i in range(self.num_samples):
            for j in range(i+1, self.num_samples):
                sample_pairs.append((sample_ids[i], sample_ids[j]))
        self.sample_pairs = sample_pairs
        self.num_pairs = len(sample_pairs)
        self.pairs_MultiIndex = pd.MultiIndex.from_tuples(self.sample_pairs,names=["sample_1","sample_2"])
        self.pairs_selected = pd.DataFrame(np.random.rand(self.num_pairs,self.num_contigs) < self.fraction_pairs,index=self.pairs_MultiIndex,columns=self.contig_ids)
        

    def compute_coal_segments(self, ts_file):
        ts = tskit.load(os.path.join(self.ts_dir, ts_file))
        node_times = ts.nodes_time
        if self.num_posterior_samples == 1:
            contig_id = ts_file.rstrip('.trees')
        else:
            contig_id = re.sub(r'_[0-9]+\.trees$', '', ts_file)
        hist,time_bins_edges = np.histogram(node_times,bins=self.num_timepoints,range=(0,self.max_time))
        hist_idxs = np.concatenate(([0],np.cumsum(hist)))

        contig_pairs_idxs = np.where(self.pairs_selected[contig_id].values)[0]
        contig_pairs = np.array(self.sample_pairs)[contig_pairs_idxs]
        coal_segments = ts.pair_coalescence_counts(sample_sets=self.sample_sets,indexes=contig_pairs,span_normalise=False)
    
        coal_segments_summary = np.zeros((self.num_pairs, len(time_bins_edges)-1), dtype=np.uint32)
        for i in range(len(time_bins_edges)-1):
            coal_segments_summary[contig_pairs_idxs,i] = np.sum(coal_segments[:,hist_idxs[i]:hist_idxs[i+1]], axis=1)
        
        return coal_segments_summary

    def compute_uncoalesced_total(self, output_dir, max_time, num_threads = 1):
        """A method to count coalescent events over time for all contigs and save results as parquet files.
        """
        os.makedirs(output_dir, exist_ok=True)
        def process_contig(ts_file):
            contig_name = os.path.basename(ts_file).rstrip('.trees')
            uncoalesced_df = self.compute_uncoalesced_contig(ts_file,max_time)
            out_path = os.path.join(output_dir, f"{contig_name}_uncoalesced.parquet")
            uncoalesced_df.to_parquet(out_path)
        print("Processing contigs to compute uncoalesced counts...")
        if num_threads > 1:
            with Pool(num_threads) as pool:
                for _ in tqdm(pool.imap(process_contig, self.ts_files), total=len(self.ts_files)):
                    pass
        else:
            for ts_file in self.ts_files:
                _ = tqdm(process_contig(ts_file), total=len(self.ts_files))

        # Aggregating the uncoalesced data across contigs and posterior samples
        print("Aggregating uncoalesced data across contigs...")
        if self.num_posterior_samples is None:
            for contig_id in tqdm(self.contig_ids):
                uncoal_file = os.path.join(output_dir, f"{contig_id}_uncoalesced.parquet")
                uncoalesced_df = pd.read_parquet(uncoal_file)
                if contig_id == self.contig_ids[0]:
                    total_uncoalesced_df = uncoalesced_df.copy()
                else:
                    total_uncoalesced_df += uncoalesced_df
            survival_df = (total_uncoalesced_df.T / total_uncoalesced_df.iloc[:,0].values).T
            survival_df.to_parquet(os.path.join(output_dir, "survival.parquet"))
        else:
            # TODO: "Implement aggregation across posterior samples if needed"
            pass
        
        self.processed_contigs = True
        return survival_df
    
    def compute_coal_rates(self, survival_df, window_size):
        # survival_time_points = survival_df.columns.values
        # survival_step = survival_time_points[1] - survival_time_points[0]
        # idxs_step = window_size // survival_step

        # negative_log_survival = -np.log(survival_df.replace(0, np.nan))
        # rates = (negative_log_survival.iloc[:,idxs_step:].values - negative_log_survival.iloc[:,:-idxs_step].values) / window_size
        # rate_times = (survival_time_points[:-idxs_step] + (window_size / 2)).astype(int)
        # rates_df = pd.DataFrame(rates,columns=rate_times,index = survival_df.index)
        
        times = survival_df.columns.astype(int)
        step_size = times[1]-times[0]
        m = window_size//step_size
        neg_log_survival = -np.log(survival_df.replace(0, np.nan))
        rates = savgol_filter(neg_log_survival,window_length=m,polyorder=1,deriv=1,delta=step_size,axis=1)
        rates_df = pd.DataFrame(rates,columns=times,index=survival_df.index)
        return rates_df
