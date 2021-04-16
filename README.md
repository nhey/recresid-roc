Recursive residuals and reversely-ordered CUSUM in futhark and python.
Is work in progress.

Futhark and python versions are validated against each other by running
`make recresid_validate` and `make roc_validate`.
Generate data sets in `/data` first, e.g. `make d-10-200-60.in`.
A `shell.nix` is provided.

Validate recursive residuals against real world data by running `make peru_recresid`.

Python versions are validated against the R reference
implementation with `make R_roc_validate` and `make R_recresid_validate`
(makes use of `r-shell.nix`).
