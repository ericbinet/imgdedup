# imgdedup

Find near-duplicate images across large collections.

## Features

- **Perceptual hashing** (pHash, dHash, wHash) for fast candidate identification
- **Pairwise scoring** with SSIM, histogram correlation, and MSE
- **Crop detection** using ORB+RANSAC homography with template matching fallback
- **Grayscale & brightness detection** for modified images
- **SQLite caching** for efficient re-runs
- **Terminal reports** with Rich formatting; optional JSON export

## Installation

Requires Python 3.11+.

```bash
pip install imgdedup
```

For RAW photo support, also install:
```bash
pip install rawpy
```

## Quick Start

```bash
# Scan and cache image hashes
imgdedup scan ~/Photos

# Find duplicates
imgdedup find

# View results
imgdedup report

# (Optionally) delete duplicates
imgdedup clean --force
```

All results are stored in `imgdedup.db` by default.

## CLI Options

```
imgdedup scan <paths>              Hash images and populate cache
imgdedup find [--find-crops]       Find candidate pairs and score
imgdedup report [--json out.json]  Display groups; optionally export JSON
imgdedup clean [--force]           Delete duplicates (dry-run by default)
```

See `imgdedup --help` for all options.

## License

MIT
