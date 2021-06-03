import "lib/github.com/diku-dk/linalg/linalg"
import "lib/github.com/nhey/lm/lm"

module linalg = mk_linalg f64
module lm = lm_f64

let nonans xs: bool =
  !(any f64.isnan xs)

let mean_abs [n] (xs: [n]f64) =
  (reduce (+) 0 (map f64.abs xs)) / (f64.i64 n)

-- R's all.equal.
let allequal target current tol =
  let xy = mean_abs (map2 (-) target current)
  let xn = mean_abs target
  in if xn > tol
     then xy/xn <= tol
     else xy <= tol

-- NOTE
-- BFAST R-package makes extensive use of the
-- strucchange R-package. However, if armadillo
-- bindings is available it will use their own
-- "strucchangeRcpp" version. In this, recursive
-- residuals differ significantly in that
-- only an absolute check is done on the equality
-- of parameters (R mostly does relative check)
-- AND it will use a simple solver rather than QR
-- whenever the condition number is less than some
-- tolerance.
--
-- This is the armadillo version, which is simply
-- an absolute difference |x - y| <= tol.
let approx_equal x y tol =
  (mean_abs (map2 (-) x y)) <= tol

-- NOTE: input cannot contain nan values
entry recresid [n][k] (X': [k][n]f64) (y: [n]f64) =
  let tol = f64.sqrt(f64.epsilon) / (f64.i64 k) -- TODO: pass tol as arg?
  let ret = replicate (n - k) 0

  -- Initialize recursion
  let model = lm.fit X'[:, :k] y[:k]
  let X1: [k][k]f64 = model.cov_params -- (X.T X)^(-1)
  let beta: [k]f64 = model.params

  let loop_body r X1r betar =
    -- Compute recursive residual
    let x = X'[:, r]
    let d = linalg.matvecmul_row X1r x
    let fr = 1 + (linalg.dotprod x d)
    let resid = y[r] - linalg.dotprod x betar
    let recresidr = resid / f64.sqrt(fr)

    -- Update formulas
    let ddT = linalg.outer d d
    -- X1r = X1r - ddT/fr
    let X1r = map2 (map2 (\x y -> x - y/fr)) X1r ddT
    -- beta = beta + X1 x * resid
    let betar = map2 (+) betar (map (linalg.dotprod x >-> (*resid)) X1r)
    in (X1r, betar, recresidr)

  -- Perform first few iterations of update formulas with
  -- numerical stability check.
  let (_, r', X1, beta, _, ret) =
    loop (check, r, X1r, betar, rank, retr) = (true, k, X1, beta, model.rank, ret)
         while check && r < n do
      let (X1r, betar, recresidr) = loop_body r X1r betar
      let retr[r-k] = recresidr

      -- Check numerical stability (rectify if unstable)
      let (check, X1r, betar, rank) =
        if check && (r+1 < n) then
          -- We check update formula value against full OLS fit
          let model = lm.fit X'[:, :r+1] y[:r+1]
          let nona = !(f64.isnan recresidr) && rank == k
                                            && model.rank == k
          let check = !(nona && approx_equal model.params betar tol)
          in (check, model.cov_params, model.params, model.rank)
       else (check, X1r, betar, rank)
      in (check, r+1, X1r, betar, rank, retr)

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

-- Map-distributed `recresid`.
entry mrecresid_nn [m][N][k] (Xs_nn: [m][N][k]f64) (ys_nn: [m][N]f64) =
  let tol = f64.sqrt(f64.epsilon) / (f64.i64 k)

  -- Initialise recursion by fitting on first `k` observations.
  let (X1s, betas, ranks) =
    map2 (\X_nn y_nn ->
            let model = lm.fit (transpose X_nn[:k, :]) y_nn[:k]
            in (model.cov_params, model.params, model.rank)
         ) Xs_nn ys_nn |> unzip3

  let num_recresids_padded = N - k
  let rets = replicate (m*num_recresids_padded) 0
             |> unflatten num_recresids_padded m

  let loop_body (r: i64) (X1: [k][k]f64) (beta: [k]f64)
                (X_nn: [N][k]f64) (y_nn: [N]f64) =
    -- Compute recursive residual
    let x = X_nn[r, :]
    let d = linalg.matvecmul_row X1 x
    let fr = 1 + (linalg.dotprod x d)
    let resid = y_nn[r] - linalg.dotprod x beta
    let recresid_r = resid / f64.sqrt(fr)
    -- Update formulas
    -- X1 = X1 - ddT/fr
    -- beta = beta + X1 x * resid
    let X1 = map2 (\d1 -> map2 (\d2 x -> x - (d1*d2)/fr) d) d X1
    let beta = map2 (+) beta (map (linalg.dotprod x >-> (*resid)) X1)
    in (X1, beta, recresid_r)

  -- Map is interchanged so that it is inside the sequential loop.
  let (_, r', X1s, betas, _, retsT) =
    loop (check, r, X1s, betas, ranks, rets_r) = (true,k,X1s,betas,ranks,rets)
      while check && r < N - 1 do
        let (checks, X1s, betas, ranks, recresids_r) = unzip5 <|
          map5 (\X1 beta X_nn y_nn rank ->
                  let (_, beta, recresid_r) = loop_body r X1 beta X_nn y_nn
                  -- Check numerical stability (rectify if unstable)
                  let (check, X1, beta, rank) =
                    -- We check update formula value against full OLS fit
                    let rp1 = r+1
                    -- NOTE We only need the transposed versions for the
                    -- first few iterations; I think it is more efficient
                    -- to transpose locally here because the matrix will
                    -- most definitely fit entirely in scratchpad memory.
                    -- Also we get to read from the array 1-strided.
                    let model = lm.fit (transpose X_nn[:rp1, :]) y_nn[:rp1]
                    -- Check that this and previous fit is full rank.
                    -- R checks nans in fitted parameters to same effect.
                    -- Also, yes it really is necessary to check all this.
                    let nona = !(f64.isnan recresid_r) && rank == k
                                                       && model.rank == k
                    let check = !(nona && approx_equal model.params beta tol)
                    -- Stop checking on all-nan ("empty") pixels.
                    let check = check && !(all f64.isnan y_nn)
                    in (check, model.cov_params, model.params, model.rank)
                  in (check, X1, beta, rank, recresid_r)
               ) X1s betas Xs_nn ys_nn ranks
        let rets_r[r-k, :] = recresids_r
        in (reduce_comm (||) false checks, r+1, X1s, betas, ranks, rets_r)

  let (_, _, retsT) =
    loop (X1s, betas, rets_r) = (X1s, betas, retsT) for r in (r'..<N) do
      let (X1s, betas, recresidrs) =
        unzip3 (map4 (loop_body r) X1s betas Xs_nn ys_nn)
      let rets_r[r-k, :] = recresidrs
      in (X1s, betas, rets_r)

  let num_checks = r' - k -- debug output
  in (retsT, num_checks)

-- Map-distributed `recresid`. There may be nan values in `ys`.
entry mrecresid [m][N][k] (X: [N][k]f64) (ys: [m][N]f64) =
  -- NOTE: the following could probably be replaced by an if-statement in
  -- the loop, which might be desirable if the loop body is fully sequentialized.
  --
  -- Rearrange `ys` so that valid values come before nans.
  let (ns, ys_nn, indss_nn) = unzip3 (map filter_nan_pad ys)
  -- Upper bound on number of non-nans
  let Nbar = i64.maximum ns
  -- Repeat this for `X`.
  let Xs_nn: *[m][Nbar][k]f64 =
    map (\j ->
           map (\i -> if i >= 0 then X[i, :] else replicate k f64.nan) indss_nn[j,:Nbar]
        ) (iota m)
  -- Subset ys
  let ys_nn = ys_nn[:,:Nbar]

  -- I expect this to optimized away.
  let _sanity_check = map (\n -> assert (n > k) true) ns
  let (retsT, num_checks) = mrecresid_nn Xs_nn ys_nn
  in (retsT, num_checks, Nbar, ns)
