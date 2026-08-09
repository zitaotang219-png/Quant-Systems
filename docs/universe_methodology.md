# Point-in-Time Crypto Universe Methodology

## Scope

The system uses a defined Crypto30 candidate universe. It is not a complete, exchange-wide historical universe and must not be interpreted as one. Candidate membership is stored in `crypto_data/binance_crypto30_daily/universe.csv`; lifecycle metadata is stored in `crypto_data/asset_master.csv` with provenance in `crypto_data/asset_master_metadata.json`.

## Initial Universe Formation

The research panel begins on 2023-01-01. An asset can first become eligible only after its lifecycle listing date and after it has accumulated the configured minimum historical bars. With the production default of 20 daily bars, the first eligible universe is formed on 2023-01-20 for assets available from the panel start.

## Listing And Delisting

An asset is in its lifecycle on dates satisfying `listing_date <= date` and, if a delisting date exists, `date <= delisting_date`. Bars before listing are classified as expected absence. A delisted asset remains eligible through its delisting date if it satisfies all other checks, and is excluded from the following date onward with reason `after_delisting`.

## History, Liquidity, And Availability

Eligibility requires the configured number of observed historical bars, complete required OHLCV values in the trailing window, and a trailing median dollar-volume value at or above the configured liquidity threshold. Production uses a zero liquidity threshold; `liquidity_sensitivity.csv` records the separate 0, USD 1m, USD 5m, and USD 10m sensitivity cases without changing that default.

The data-quality report distinguishes true missing bars from expected lifecycle absence. A true missing bar is a missing observation on a market-calendar date inside an asset's lifecycle. Dates before listing and after delisting are not counted as missing market data.

## Known Limitations

- The Crypto30 candidate list is manually defined and may omit historically listed assets outside that list.
- Lifecycle metadata is manually curated from exchange metadata and requires periodic source refresh and independent verification.
- The data uses a shared daily calendar; exchange outages and asset-specific trading suspensions require separate market-calendar metadata if they must be distinguished from true missing data.
- A static reference-weight benchmark is used when historical point-in-time market-cap data is unavailable.
