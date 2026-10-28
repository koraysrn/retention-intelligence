# Data fixtures

`ecommerce_sample.csv` is a **synthetic** 1,200-row dataset that follows the
exact 27-column schema of `ecommerce_data.csv`.

- Generator: `python -m scripts.generate_fixture` (seed 42)
- License: CC0-1.0 (synthetic; no real people or customer records)
- Checksum: `ecommerce_sample.sha256`
- Recommended as-of date: `2025-10-02` (dataset end 2025-12-31 minus 90 days)
- Outcome window: `2025-10-02 < order_date <= 2025-12-31`

The fixture is committed so that a clean clone can run
`make data && make pipeline && make test` without network access. To use a
licensed real dataset, set `DATA_URL` when running `python -m scripts.download_data`.
