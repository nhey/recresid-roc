import "lib/github.com/diku-dk/linalg/linalg"
import "lib/github.com/nhey/ols/ols"

module linalg = mk_linalg f64
module ols = mk_ols f64

let nonans xs: bool =
  !(any f64.isnan xs)

let nan_to_num [n] (num: f64) (xs: [n]f64): [n]f64 =
  map (\x -> if f64.isnan x then num else x) xs

-- Dotproduct ignoring nans.
let dotprod_nan [n] (xs: [n]f64) (ys: [n]f64): f64 =
  reduce (+) 0 (map2 (\x y -> if f64.isnan y then 0 else x*y) xs ys)

-- From helpers.fut:
-- Dotproduct filtering terms based on third vector (filter).
let dotprod_filt [n] (mask: [n]f64) (xs: [n]f64) (ys: [n]f64) : f64 =
  f64.sum (map3 (\v x y -> x * y * if (f64.isnan v) then 0.0 else 1.0) mask xs ys)

let mvmul_filt [n][m] (mask: [m]f64) (xss: [n][m]f64) (ys: [m]f64) =
  map (dotprod_filt mask ys) xss

let scanExc 't [n] (op: t->t->t) (ne: t) (arr : [n]t) : [n]t =
  scan op ne <| map (\i -> if i>0 then arr[i-1] else ne) (iota n)

let non_nan_inds arr =
  let flags = map (\x -> if !(f64.isnan x) then 1 else 0) arr
  let inds_acc = scan (+) 0 flags
  let mapping = map2 (\f i -> f*i - 1) flags inds_acc
  let num_non_nan = last inds_acc
  in scatter (replicate num_non_nan 0) mapping (indices arr)

-- Translation of the C code here:
-- https://en.wikipedia.org/wiki/Machine_epsilon
let f64_eps =
  let sf64 = 1.0f64
  let su64 = f64.to_bits sf64
  let nearest = su64 + 1u64
  in (f64.from_bits nearest) - sf64

entry recresid [n][k] (bsz: i64) (X: [n][k]f64) (y: [n]f64) =
  let tol = f64.sqrt(f64_eps) / (f64.i64 k) -- TODO: pass tol as arg?
  -- Compute actual inner size and non nan indices.
  -- NOTE: this could be replaced by an if-statement in the loop,
  -- which is probably desirable if the loop body is fully sequentialized.
  -- let inds = non_nan_inds y
  -- let n' = length inds
  -- let _sz = assert (n' - k > 0) 0
  let ret = replicate (n - k) 0
  
  -- initialize recursion
  let model = ols.fit bsz X[:k, :] y[:k]
  let X1: [k][k]f64 = model.cov_params -- (X.T X)^(-1)
  let beta: [k]f64 = nan_to_num 0 model.params

  let kk = k*k

  let loop_body r X1r betar =
    -- Compute recursive residual
    let x = X[r, :]
    let d = linalg.matvecmul_row X1r x
    let fr = 1 + (linalg.dotprod x d)
    let resid = y[r] - dotprod_nan x betar
    let recresidr = resid / f64.sqrt(fr)

    -- Update formulas
    let ddT = linalg.outer d d
    -- X1r = X1r - ddT/fr
    let X1r = map2 (\x y -> x - y/fr)
                   (flatten X1r :> [kk]f64)
                   (flatten ddT :> [kk]f64) |> unflatten k k
    -- beta = beta + X1 x * resid
    let betar = map2 (+) betar (map (dotprod_nan x >-> (*resid)) X1r)
    in (X1r, betar, recresidr)

  -- TODO: how do we tell the futhark compiler to perform upper bound
  -- on checks over all pixels? E.g. we want to keep executing this loop
  -- until all are done checking?
  -- Will probably be clearer once outer map is distributed.
  --
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
          let model = ols.fit bsz X[:r+1, :] y[:r+1]
          let nona = nonans(betar) && nonans(model.params)
          let allclose = map2 (-) model.params betar
                         |> all (\x -> f64.abs x <= tol)
          in (!(nona && allclose), model.cov_params, nan_to_num 0 model.params)
       else (check, X1r, betar)
      in (check, r+1, X1r, betar, retr)

  -- Perform remaining iterations without check.
  -- The loop is split to avoid thread divergence when this
  -- function is mapped over an array.
  let (_, _, ret) =
    loop (X1r, betar, retr) = (X1, beta, ret) for r in (r'..<n) do
      let (X1r, betar, recresidr) = loop_body r X1r betar
      let retr[r-k] = recresidr
      in (X1r, betar, retr)

  -- let num_check = r' - k
  -- in (ret, num_check)
  in ret
