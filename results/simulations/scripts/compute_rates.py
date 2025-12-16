#!/usr/bin/env python3
"""
Compute coalescence rates from tree sequences.
"""

import argparse


def main():
    parser = argparse.ArgumentParser(description="Compute coalescence rates from tree sequences")
    parser.add_argument("--tree_path", required=True, help="Path to tree sequence directory")
    parser.add_argument("--output_path", required=True, help="Output directory for rates data")
    parser.add_argument("--output_name", required=True, help="Output filename for rates parquet file")
    parser.add_argument("--max_time", type=int, required=True, help="Maximum time for rate calculation")
    parser.add_argument("--window_size", type=int, required=True, help="Window size for rate calculation")
    
    args = parser.parse_args()
    
    # TODO: Implement rates computation logic
    print(f"Computing rates from trees at {args.tree_path}")
    print(f"Output path: {args.output_path}")
    print(f"Output name: {args.output_name}")
    print(f"Max time: {args.max_time}")
    print(f"Window size: {args.window_size}")


if __name__ == "__main__":
    main()
