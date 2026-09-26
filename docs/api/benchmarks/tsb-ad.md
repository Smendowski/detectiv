# TSB-AD API

## CSV Loader

::: detectiv.benchmarks.tsb_ad.load_tsb_ad_csv

## Collections

::: detectiv.benchmarks.tsb_ad.TSBADCollection

::: detectiv.benchmarks.tsb_ad.TSBADCollectionLoader

## Upstream Evaluation

::: detectiv.benchmarks.tsb_ad.TSBADAdapter

Pass this evaluator to `detectiv.callbacks.MetricsCallback`; the callback reads
labels from the completed reconstruction report and qualifies every returned
metric as `<plan>.<propagation>.<metric>`.

::: detectiv.callbacks.MetricsCallback
