# FlyMind Data Directory

This directory stores raw and processed connectome data files for the FlyMind project.

## Directory Structure

```
data/
├── raw/         # Unmodified raw JSON/CSV responses or dumps from neuPrint/FlyWire
├── processed/   # Cleaned, validated subcircuit graphs in sparse format (JSON, Parquet)
└── README.md    # Data provenance and licensing
```

## Data Provenance & Attribution

All connectome data utilized in FlyMind originates from:
- **Janelia FlyEM Project**: Scheffer et al. (2020) *A connectome and analysis of the adult Drosophila central brain*.
- **Hosting Service**: [neuprint.janelia.org](https://neuprint.janelia.org)
- **License**: Creative Commons Attribution 4.0 International ([CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/)).

## Storage Policy

- **No Massive Raw Dumps**: Do not commit multi-gigabyte whole-brain database dumps to this repository.
- **Reproducible Extracts**: Target subcircuits (such as the Central Complex navigation loop) are extracted via `flymind.connectome.query` and cached in `data/processed/` as compact, sparse graph representations (< 5 MB).
