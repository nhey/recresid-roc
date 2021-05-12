Recursive residuals and reversely-ordered CUSUM in futhark and python.
Map-distributed versions are provided; they produce satisfactory results
(correct within tolerance), but have yet to be optimised.

Futhark and python versions are validated against each other by running
`make validate_recresid` and `make validate_roc`.
Generate data sets in `/data` first, e.g. `make d-10-200-60.in`.
A `shell.nix` is provided.

Python versions are validated against the R reference
implementation with `make R_roc_validate` and `make R_recresid_validate`
(makes use of `r-shell.nix`).
