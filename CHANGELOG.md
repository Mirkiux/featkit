# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
