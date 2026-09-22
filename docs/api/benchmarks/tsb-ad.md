# TSB-AD API

## CSV Loader

::: detectiv.benchmarks.tsb_ad.TSBADCsvLoader

## Collections

::: detectiv.benchmarks.tsb_ad.TSBADCollection

::: detectiv.benchmarks.tsb_ad.TSBADDataset

::: detectiv.benchmarks.tsb_ad.TSBADCollectionLoader

## Upstream Evaluation

::: detectiv.benchmarks.tsb_ad.TSBADAdapter

::: detectiv.benchmarks.tsb_ad.TSBADEvaluator

Pass this evaluator to `detectiv.callbacks.MetricsCallback`; the callback reads
labels from the completed reconstruction report and qualifies every returned
metric by scoring plan, propagation, and series.

::: detectiv.callbacks.MetricsCallback
