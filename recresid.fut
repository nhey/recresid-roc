import "lib/github.com/diku-dk/linalg/linalg"
import "lib/github.com/nhey/ols/ols"
import "lm/lm"

import "lm/linpack"
module linpack = linpack_d

module linalg = mk_linalg f64
module ols = mk_ols f64
module lm = lm_f64

let nonans xs: bool =
  !(any f64.isnan xs)

let nan_to_num [n] (num: f64) (xs: [n]f64): [n]f64 =
  map (\x -> if f64.isnan x then num else x) xs

-- Dotproduct ignoring nans.
let dotprod_nan [n] (xs: [n]f64) (ys: [n]f64): f64 =
  reduce (+) 0 (map2 (\x y -> if f64.isnan y then 0 else x*y) xs ys)

-- NOTE: input cannot contain nan values
entry recresid [n][k] (X': [k][n]f64) (y: [n]f64) =
  let tol = f64.sqrt(f64.epsilon) / (f64.i64 k) -- TODO: pass tol as arg?
  let ret = replicate (n - k) 0

  -- Initialize recursion
  let model = lm.fit X'[:, :k] y[:k]
  let X1: [k][k]f64 = model.cov_params -- (X.T X)^(-1)
  let beta: [k]f64 = nan_to_num 0 model.params

  let loop_body r X1r betar =
    -- Compute recursive residual
    let x = X'[:, r]
    let d = linalg.matvecmul_row X1r x
    let fr = 1 + (linalg.dotprod x d)
    let resid = y[r] - dotprod_nan x betar
    let recresidr = resid / f64.sqrt(fr)

    -- Update formulas
    let ddT = linalg.outer d d
    -- X1r = X1r - ddT/fr
    let X1r = map2 (map2 (\x y -> x - y/fr)) X1r ddT
    -- beta = beta + X1 x * resid
    let betar = map2 (+) betar (map (dotprod_nan x >-> (*resid)) X1r)
    in (X1r, betar, recresidr)

  -- Perform first few iterations of update formulas with
  -- numerical stability check.
  let (_, r', X1, beta, ret) =
    loop (check, r, X1r, betar, retr) = (true, k, X1, beta, ret)
         while check && r < n do
      let (X1r, betar, recresidr) = loop_body r X1r betar
      let retr[r-k] = recresidr

      -- Check numerical stability (rectify if unstable)
      let (check, X1r, betar) =
        if check && (r+1 < n) then
          -- We check update formula value against full OLS fit
          let model = lm.fit X'[:, :r+1] y[:r+1]
          let nona = model.rank == k
          let allclose = map2 (-) model.params betar
                         |> all (\x -> f64.abs x <= tol)
          in (!(nona && allclose), model.cov_params, nan_to_num 0 model.params)
       else (check, X1r, betar)
      in (check, r+1, X1r, betar, retr)

  -- Perform remaining iterations without check.
  let (_, _, ret) =
    loop (X1r, betar, retr) = (X1, beta, ret) for r in (r'..<n) do
      let (X1r, betar, recresidr) = loop_body r X1r betar
      let retr[r-k] = recresidr
      in (X1r, betar, retr)

  let num_checks = r' - k -- debug output
  in (ret, num_checks)


-- Helpers for lifted recresid.
-- Dotproduct filtering terms based on third vector (filter).
let dotprod_filt [n] (mask: [n]f64) (xs: [n]f64) (ys: [n]f64) : f64 =
  f64.sum (map3 (\v x y -> x * y * if (f64.isnan v) then 0.0 else 1.0) mask xs ys)

let mvmul_filt [n][m] (mask: [m]f64) (xss: [n][m]f64) (ys: [m]f64) =
  map (dotprod_filt mask ys) xss

-- Rearrange so that values, true under predicate `p`, come first.
-- Pad to original length with dummy values.
-- Returns number of non-dummy values,
-- array of values padded to original legnth and indices
-- mapping each value to its location in the original array.
let filterPadWithKeys [n] 't
           (p : (t -> bool))
           (dummy : t)
           (arr : [n]t) : (i64, [n]t, [n]i64) =
  let tfs = map (\a -> if p a then 1i64 else 0i64) arr
  let isT = scan (+) 0i64 tfs
  let i   = last isT
  let inds= map2 (\a iT -> if p a then iT - 1 else -1i64) arr isT
  let rs  = scatter (replicate n dummy) inds arr
  let ks  = scatter (replicate n (-1i64)) inds (iota n)
  in (i, rs, ks)

let filter_nan_pad = filterPadWithKeys ((!) <-< f64.isnan) f64.nan

-- Map-distributed `recresid`. There may be nan values in `ys`.
entry mrecresid [m][N][k] (X: [N][k]f64) (ys: [m][N]f64) =
  let tol = f64.sqrt(f64.epsilon) / (f64.i64 k)

  -- NOTE: the following could probably be replaced by an if-statement in
  -- the loop, which might be desirable if the loop body is fully sequentialized.
  --
  -- Rearrange `ys` so that valid values come before nans.
  let (ns, ys_nn, indss_nn) = unzip3 (map filter_nan_pad ys)
  -- Repeat this for `X`.
  let Xs_nn: *[m][N][k]f64 =
    map (\j ->
           map (\i -> if i >= 0 then X[i, :] else replicate k f64.nan) indss_nn[j]
        ) (iota m)

  -- Initialise recursion by fitting on first `k` observations.
  let _sanity_check = map (\n -> assert (n > k) true) ns
  -- TODO: Fuse with Xs_nn loop above? `ys_nn` is same order as `indss_nn`.
  let (X1s, betas, ranks) = map2 (\X_nn y_nn ->
                                    let model = lm.fit (transpose X_nn[:k, :]) y_nn[:k]
                                    in (map (nan_to_num 0) model.cov_params, nan_to_num 0 model.params, model.rank)
                                 ) Xs_nn ys_nn |> unzip3

  let num_recresids_padded = N - k
  let rets = replicate (m*num_recresids_padded) 0
             |> unflatten num_recresids_padded m

  -- Map is interchanged so that it is inside the sequential loop.
  let (_, r', X1s, betas, _, retsT) =
    loop (check, r, X1rs, betars, ranks, retrs) = (true, k, X1s, betas, ranks, rets)
         while check && r < N - 1 do
      let (checks, X1rs, betars, ranks, recresidrs) = unzip5 <|
        map5 (\X1r betar X_nn y_nn rank ->
                -- Compute recursive residual
                let x = X_nn[r, :]
                let d = mvmul_filt x X1r x
                let fr = 1 + (dotprod_nan x d)
                let resid = y_nn[r] - dotprod_nan x betar
                let recresidr = resid / f64.sqrt(fr)

                -- Update formulas
                let ddT = linalg.outer d d
                -- X1r = X1r - ddT/fr
                let X1r = map2 (map2 (\x y -> x - y/fr)) X1r ddT
                -- beta = beta + X1 x * resid
                let betar = map2 (+) betar (map (dotprod_nan x >-> (*resid)) X1r)

                -- Check numerical stability (rectify if unstable)
                let (check, X1r, betar, rank) =
                  -- We check update formula value against full OLS fit
                  let rp1 = r+1
                  -- TODO dont do this transpose each time...
                  let model = lm.fit (transpose X_nn[:rp1, :]) y_nn[:rp1]
                  -- Check that this and previous fit is full rank.
                  -- R checks nans in fitted parameters to same effect.
                  let nona = rank == k && model.rank == k
                  let allclose = map2 (-) model.params betar
                                 |> all (\x -> f64.abs x <= tol)
                  -- TODO make lm not ouput nans but zeros...
                  in (!(nona && allclose), map (nan_to_num 0) model.cov_params, nan_to_num 0 model.params, model.rank)
                in (check, X1r, betar, rank, recresidr)
             ) X1rs betars Xs_nn ys_nn ranks
      let retrs[r-k, :] = recresidrs
      in (reduce_comm (||) false checks, r+1, X1rs, betars, ranks, retrs)

  let (_, _, retsT) =
    loop (X1rs, betars, retrs) = (X1s, betas, retsT) for r in (r'..<N) do
      let (X1rs, betars, recresidrs) = unzip3 <|
        map4 (\X1r betar X_nn y_nn ->
                -- Compute recursive residual
                let x = X_nn[r, :]
                let d = mvmul_filt x X1r x
                let fr = 1 + (dotprod_nan x d)
                let resid = y_nn[r] - dotprod_nan x betar
                let recresidr = resid / f64.sqrt(fr)

                -- Update formulas
                let ddT = linalg.outer d d
                -- X1r = X1r - ddT/fr
                let X1r = map2 (map2 (\x y -> x - y/fr)) X1r ddT
                -- beta = beta + X1 x * resid
                let betar = map2 (+) betar (map (dotprod_nan x >-> (*resid)) X1r)
                in (X1r, betar, recresidr)
             ) X1rs betars Xs_nn ys_nn
      let retrs[r-k, :] = recresidrs
      in (X1rs, betars, retrs)

  let num_checks = r' - k -- debug output
  in (retsT, num_checks)

entry mrecresid1 X' y = (mrecresid (transpose X') y).0 |> transpose
