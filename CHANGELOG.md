# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.2] - 2026-06-30

### Fixed
- `FREQ` operator now counts only periods where the value is non-null **and strictly greater than 0** (previously counted any non-null value).
- `XM` operator now returns `1` only when **every** period in the time window has a non-null and strictly positive value, `0` otherwise (previously returned a raw count identical to FREQ). Both the SQL and PySpark generators are updated.

## [0.4.1] - 2026-06-09

### Fixed
- CI: auto-tag workflow now uses a PAT (`RELEASE_TOKEN`) to push tags so that `publish.yml` is triggered correctly (`fix(ci)`)

## [0.4.0] - 2026-06-09

### Added
- Ratio/percentage features (`RatioPivotedColumn`, `RatioSpaceBuilder`): for every pivot combination with at least one non-`None` categorical value, a `numerator / NULLIF(denominator, 0)` column is generated for each proper marginal projection of that combination. Controlled by `FeatureStoreConfig.include_ratios` (default `True`, requires `include_marginals=True`). (`feat(ratio)`)
- `verbose` parameter on `AdapterDomainResolver` and `AdapterCombinationResolver`: when `True`, the generated `SELECT DISTINCT` SQL is emitted at `DEBUG` level before execution. `FeatureStorePipeline` forwards `cfg.verbose` to the combination resolver automatically. (`feat(domain-resolver)`)

## [0.3.0] - 2026-06-08

### Added
- `AdapterCombinationResolver` — replaces per-field `SELECT DISTINCT` queries with a single multi-column query returning only observed combinations (`feat(builders)`)
- `verbose` logging option on `PivotSpaceBuilder`, `DistributionalSpaceBuilder`, and `TemporalSpaceBuilder`, configurable via `FeatureStoreConfig` (`feat(config)`)

### Fixed
- Marginal fields no longer contribute their name to pivot column names; e.g. `SUM__amount__channel__region_north` → `SUM__amount__region_north` (`fix(layer2)`)

## [0.2.0] - 2026-06-02

### Added
- Execution layer with adapter-based domain resolution (`feat(execution)`)

### Fixed
- Lazy-import `AdapterDomainResolver`; added `pandas` to dev dependencies
