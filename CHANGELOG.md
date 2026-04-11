# ChangeLog

## [0.2.2] - 2026-04-11

### Added

- QUANTILE

## [0.2.1] - 2026-04-06

### Fix

- Fixed an internal logic bug in `Context::end` where `_end == 0` incorrectly returned `0` instead of yielding the expected `total` array dimension.
- Changed Context data truncation (`end` limit) to execute within the closure of each parallel grouped chunk (via `ctx.end()`) rather than globally shrinking arrays ahead of time (`align_end()`). This prevents group size corruption and misaligned cross-sectional indexing when `groups > 1`.
- Bound internal algorithm iterations and `SkipNanWindow` initializations dynamically to the `end` limit on a per-chunk basis while preserving absolute `start` index zero-alignments.
- Deprecated and removed legacy `align_end` and `align_end_mut` functions entirely to enforce correct boundary lifecycle patterns across algorithms.


## [0.2.0] - 2026-02-26

### Change

- Naming Rules for algorithms
    - Functions without prefix means it is a rolling window operation
    - Functions with prefix `CC_` means it is a cross-commodity/cross-security/cross-group operation

### Add 

- Context now have a `end` option, which can be used to specify the end index of the calculation. usually used in iterative back test to improve some performance.


## [0.1.3] - 2026-02-22


@tic-top had verified the correctness of the computation, it's a great help for us to improve the quality of the alpha library. Thanks @tic-top!

## [0.1.2] - 2026-02-04

### Added

- TS_CORR @zhaojun6969

## [0.1.1] - 2026-02-02

### Added

- BINS @zhaojun6969
- FRET @zhaojun6969 
- INTERCEPT @zhaojun6969
- NEUTRALIZE @zhaojun6969
- REGBETA
- REGRESI
- SUMIF

See [algo.md](python/alpha/algo.md) for details.

Thanks @zhaojun6969, our first contributor!
