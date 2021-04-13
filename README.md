Recursive residuals and reversely-ordered CUSUM in futhark.
Is work in progress.

Validate by running
`make recresid_validate` and `make roc_validate`. Generate data sets in `/data` first, e.g. `make d-10-200-60.in`.
A `shell.nix` is provided.

Python version of ROC is validated against the R reference
implementation with `make R_validate` (use `r-shell.nix`).
