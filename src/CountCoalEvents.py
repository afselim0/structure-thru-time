#%%
#!/usr/bin/env python3
import numpy as np
import pandas as pd
import os
from pathlib import Path
from tqdm import tqdm
import re
import tskit
from pathos.multiprocessing import ProcessingPool as Pool

class CountCoalEvents:
    def __init__(self,
                 ts_dir: str,
                 sample_ids: list = None,
                 num_posterior_samples: int = 1,
                 max_time: float = None,
                 num_timepoints: int = 100,
                 fraction_samples: float = 1.0):
        """A class for counting coalescent events from a set of tree sequences and computing pairwise coalescent rates

        Args:
            ts_dir (str): A directory containing tree sequences
            sample_ids (list, optional): List of IDs for samples to compute pairwise coalescent rates for. Defaults to None.
            num_posterior_samples (int, optional): The number of posterior ARG samples used in the case of Bayesian ARG inference. Defaults to None.
            max_time (float, optional): Maximum time in number of generations ago. Defaults to None.
            num_timepoints (int, optional): Number of time points to compute rates for. Rates are computed in a constant grid from zero to max_time. Defaults to 100.
            fraction_samples (float, optional): Fraction of sample pairs to chosen from each contig to count coalescent events. Defaults to 1.0.
        """
        
        self.ts_dir = ts_dir
        self.num_timepoints = num_timepoints
        self.fraction_samples = fraction_samples

        print("preprocessing tree sequence files...")
        # basic input validation
        assert os.path.isdir(ts_dir), f"ts_dir is not a valid directory: {ts_dir}"

        # locating tree sequence files and counting contigs and posterior samples
        self.ts_files = [file.name for file in Path(ts_dir).glob("*.trees")]
        if len(self.ts_files) == 0:
            raise ValueError(f"No .trees files found in directory: {ts_dir}")
        self.ts_files.sort()
        if num_posterior_samples == 1:
            self.num_posterior_samples = 1
            self.num_contigs = len(self.ts_files)
            self.contig_ids = [ts_file.removesuffix(".trees") for ts_file in self.ts_files]
            self.contig_lengths = np.array([tskit.load(os.path.join(ts_dir, contig_id + ".trees")).sequence_length for contig_id in tqdm(self.contig_ids,total=self.num_contigs)]).astype(np.uint64)
        else:
            self.num_posterior_samples = num_posterior_samples
            self.num_contigs = len(self.ts_files) // num_posterior_samples
            self.contig_ids = np.unique([re.sub(r'_[0-9]+\.trees$', '', contig) for contig in self.ts_files]) ## Must have the format {ContigID_PosteriorSampleID.trees}
            if len(self.ts_files) % num_posterior_samples != 0:
                raise ValueError(f"Number of .trees files ({len(self.ts_files)}) is not divisible by num_posterior_samples ({num_posterior_samples})")
            self.contig_lengths = np.array([tskit.load(os.path.join(ts_dir, contig_id + "_0.trees")).sequence_length for contig_id in tqdm(self.contig_ids,total=self.num_contigs)]).astype(np.uint64)

        tot_contig_length = np.sum(self.contig_lengths)
        if tot_contig_length < 2**32:
            self.datatype = np.uint32
        else:
            self.datatype = np.uint64
        # processing sample data
        ts = tskit.load(os.path.join(ts_dir, self.ts_files[0]))
        self.sample_sets = [[sample_id.astype(np.int32)] for sample_id in ts.samples()]
        if sample_ids is None:
            sample_ids = ts.samples()
            self.num_samples = len(sample_ids)
            self.sample_ids = sample_ids
        else:
            sample_ids = np.array(sample_ids)
            sample_ids = np.sort(sample_ids)
            self.num_samples = len(sample_ids)
            self.sample_ids = sample_ids

        if max_time is None:
            max_time = np.quantile(ts.nodes_time, 0.95)
        self.max_time = max_time

        # creating pairs of samples
        sample_pairs = []
        for i in range(self.num_samples):
            for j in range(i+1, self.num_samples):
                sample_pairs.append((sample_ids[i].astype(np.int32), sample_ids[j].astype(np.int32)))
        self.sample_pairs = sample_pairs
        self.num_pairs = len(sample_pairs)
        self.pairs_MultiIndex = pd.MultiIndex.from_tuples(self.sample_pairs,names=["sample_1","sample_2"])
        self.samples_selected = pd.DataFrame(np.random.rand(self.num_samples,self.num_contigs) < self.fraction_samples,index=self.sample_ids,columns=self.contig_ids)
        self.pair_to_idx = {pair: k for k, pair in enumerate(self.sample_pairs)}
        

    def count_coal_segments(self, ts_file):
        """For a given tree sequence, computes the total span of genomic segments that coalesce within specified time bins for all selected sample pairs.

        Args:
            ts_file (str): The tree sequence file name.

        Returns:
            np.NDArray: Each row corresponds to a sample pair, and each column corresponds to a time bin. The values represent the total span of genomic segments that coalesce within the respective time bin for the sample pair.
        """
        # Find the contig ID associated with the tree sequence file
        ts = tskit.load(os.path.join(self.ts_dir, ts_file))
        if self.num_posterior_samples == 1:
            contig_id = ts_file.removesuffix(".trees")
        else:
            contig_id = re.sub(r'_[0-9]+\.trees$', '', ts_file)

        # Find the pairs of samples selected for this contig
        s = self.samples_selected[contig_id]
        active_samples = s[s].index.values
        if len(active_samples) == 0:
            return np.zeros((self.num_pairs, self.num_timepoints), dtype=self.datatype)
        
        if len(active_samples) < ts.num_samples:
            ts = ts.simplify(samples=active_samples)

        # Matching idxs
        pairs = []
        pair_idxs = []
        for i, sample_1 in enumerate(active_samples):
            for j in range(i + 1, len(active_samples)):
                sample_2 = active_samples[j]
                pairs.append((i, j))
                pair_idxs.append(self.pair_to_idx[(sample_1, sample_2)])
        pair_idx = np.array(pair_idxs)
        
        # Counting coal events
        coal_segments_tot = np.zeros((self.num_pairs, self.num_timepoints), dtype=self.datatype)
        coal_segments = ts.pair_coalescence_counts(sample_sets=[[i] for i in range(len(active_samples))],indexes=pairs,span_normalise=False,time_windows=np.linspace(0,self.max_time,self.num_timepoints+1)).astype(self.datatype)
        coal_segments_tot[pair_idx, :] = coal_segments

        return coal_segments_tot
    
    def count_coal_segments_contigs(self, ts_files):
        """Counts coalescent segments for a set of tree sequences.

        Args:
            ts_files (List): A list of file names for tree sequences to be used.

        Returns:
            np.NDArray: Each row corresponds to a sample pair, and each column corresponds to a time bin. The values represent the total span of genomic segments that coalesce within the respective time bin for the sample pair.
        """
        for ts_file in tqdm(ts_files, total=len(ts_files), desc=f"Computing coalescent segments for contigs"):
            coal_segments = self.count_coal_segments(ts_file)
            if ts_file == ts_files[0]:
                total_coal_segments = coal_segments
            else:
                total_coal_segments += coal_segments
        return total_coal_segments
    
    def compute_coal_time_pdf(self, num_threads = 1, data_output = None):
        """Computes an estimate of the PDF of pairwise coalescent rates

        Args:
            num_threads (int, optional): Number of parallel threads to use. Defaults to 1.
            data_output (str, optional): An optional directory name to store intermediate files for parallel compute. Defaults to None.

        Returns:
            pd.DataFrame: A dataframe where each row corresponds to a sample pair, and each column corresponds to a time bin. The values represent the estimated PDF of coalescent events within the respective time bin for the sample pair.
        """
        if num_threads > len(self.ts_files):
            num_threads = len(self.ts_files)

        if num_threads == 1:
            coal_segments_total = self.count_coal_segments_contigs(self.ts_files)

        else:
            assert data_output is not None, "data_output directory must be specified when using multiple threads."
            os.makedirs(data_output, exist_ok=True)
            # Save the file as a random integer that is returned
            def process_chunk(ts_files_chunk, chunk_id):
                coal_segments_chunk = self.count_coal_segments_contigs(ts_files_chunk)
                chunk_file = os.path.join(data_output, f"coal_segments_chunk_{chunk_id}.npy")
                np.save(chunk_file, coal_segments_chunk)
                return
            
            chunks = np.array_split(self.ts_files, num_threads)

            # Pathos parallelization
            def _process_chunk_wrapper(args):
                return process_chunk(*args)
            pool = Pool(nodes=num_threads)
            results = pool.map(
                _process_chunk_wrapper,
                [(ts_files_chunk, chunk_id)
                for chunk_id, ts_files_chunk in enumerate(chunks)])

            # Aggregating the results from chunk files
            for chunk_id in range(len(chunks)):
                chunk_file = os.path.join(data_output, f"coal_segments_chunk_{chunk_id}.npy")
                coal_segments_chunk = np.load(chunk_file)
                if chunk_id == 0:
                    coal_segments_total = coal_segments_chunk
                else:
                    coal_segments_total += coal_segments_chunk
                os.remove(chunk_file)  # Clean up chunk file after loading
            os.rmdir(data_output)  # Remove the output directory if empty
        time_bins_edges = np.linspace(0, self.max_time, self.num_timepoints + 1)
        pdf_df = pd.DataFrame(coal_segments_total,columns=time_bins_edges[:-1],index = self.pairs_MultiIndex)
        pairs_selected = pd.DataFrame(
            np.logical_and(self.samples_selected.values[:,None,:], self.samples_selected.values[None,:,:])
            .reshape(self.num_samples**2,self.num_contigs),
            index=pd.MultiIndex.from_product([self.sample_ids, self.sample_ids], names=["sample_1", "sample_2"]),
            columns=self.contig_ids
        )
        mask = (pairs_selected.index.get_level_values("sample_1") < pairs_selected.index.get_level_values("sample_2"))
        pairs_selected = pairs_selected[mask]

        total_segments_per_pair = (pairs_selected*self.contig_lengths).values.sum(axis=1).astype(self.datatype)
        pdf_df = pdf_df / (total_segments_per_pair.reshape(-1,1)*self.num_posterior_samples)
        return pdf_df
# %%
