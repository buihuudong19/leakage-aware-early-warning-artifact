# Supplementary files

`rq1b_tuned_ranking_flip_xgboost305_linux.csv` is the learner comparison with nested tuning (output of the `rq1b`
stage) from an earlier run (21 October 2026) of an earlier version of the code. It differs from the reported run in
three respects: XGBoost 3.0.5, scikit-learn 1.5.2 and pandas 2.2.3 on Linux (Python 3.13), against XGBoost 3.4.1,
scikit-learn 1.9.1 and pandas 3.0.6 on Windows (Python 3.14); an XGBoost grid with depths 2, 3 and 5, against
1, 2, 3, 5 and 8; and a logistic-regression grid with six penalties from C = 0.001 to 1, against eleven from 1e-4
to 10. The file therefore shows the combined effect of these changes on the reference-protocol difference between
XGBoost and logistic regression; it does not isolate the library version. It is reported in Table S3 of the
supplementary material of the article.

Columns: condition (`shortcuts`), metric, mean of XGBoost (`mean_a`), mean of logistic regression (`mean_b`), paired
difference XGBoost minus logistic regression with its 95% block-bootstrap interval, and the share of windows won by
XGBoost.
