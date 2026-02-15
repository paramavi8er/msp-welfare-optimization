# Sample Data

These files are **synthetic** — generated to have similar statistical
properties to the real data (means, standard deviations, autocorrelation
structure) but are NOT actual Agmarknet or IMD observations.

Use them to verify the pipeline runs end-to-end:

```bash
python src/02_preprocessing.py --synthetic
python src/03_forecasting.py
```

For actual replication, download the real data following instructions
in `data/README.md`.
