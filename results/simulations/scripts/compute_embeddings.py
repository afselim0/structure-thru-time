#!/usr/bin/env python3
"""
Compute embeddings from coalescence rates.
"""

import argparse


def main():
    parser = argparse.ArgumentParser(description="Compute embeddings from coalescence rates")
    parser.add_argument("--input", required=True, help="Path to rates parquet file")
    parser.add_argument("--output_path", required=True, help="Output directory for embeddings data")
    parser.add_argument("--output_name", required=True, help="Output filename for embeddings parquet file")
    
    args = parser.parse_args()
    
    # TODO: Implement embeddings computation logic
    print(f"Computing embeddings from {args.input}")
    print(f"Output path: {args.output_path}")
    print(f"Output name: {args.output_name}")


if __name__ == "__main__":
    main()
