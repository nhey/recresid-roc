import "lib/github.com/diku-dk/linalg/linalg"
import "lib/github.com/nhey/ols/ols"

module linalg = mk_linalg f64
module ols = mk_ols f64

let nonans xs: bool =
  !(any f64.isnan xs)

let nan_to_num [n] (num: f64) (xs: [n]f64): [n]f64 =
  map (\x -> if f64.isnan x then num else x) xs

let nandot [n] (xs: [n]f64) (ys: [n]f64): f64 =
  reduce (+) 0 (map2 (\x y -> if f64.isnan y then 0 else x*y) xs ys)

-- Translation of the C code here:
-- https://en.wikipedia.org/wiki/Machine_epsilon
let f64_eps =
  let sf64 = 1.0f64
  let su64 = f64.to_bits sf64
  let nearest = su64 + 1u64
  in (f64.from_bits nearest) - sf64

entry recresid [n][k] (bsz: i64) (X: [n][k]f64) (y: [n]f64) =
  let tol = f64.sqrt(f64_eps) / (f64.i64 k) -- TODO: pass tol as arg?
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
    let resid = y[r] - nandot x betar
    let recresidr = resid / f64.sqrt(fr)

    -- Update formulas
    let ddT = linalg.outer d d
    -- X1r = X1r - ddT/fr
    let X1r = map2 (\x y -> x - y/fr)
                   (flatten X1r :> [kk]f64)
                   (flatten ddT :> [kk]f64) |> unflatten k k
    -- beta = beta + X1 x * resid
    let betar = map2 (+) betar (map (nandot x >-> (*resid)) X1r)
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
          let model = ols.fit bsz X[:r+1, :] y[:r+1]
          let nona = nonans(betar) && nonans(model.params)
          let allclose = map2 (-) model.params betar
                         |> all (\x -> f64.abs x <= tol)
          in (!(nona && allclose), model.cov_params, nan_to_num 0 model.params)
       else (check, X1r, betar)
      in (check, r+1, X1r, betar, retr)

  -- Perform remaining iterations without check.
  -- The loop is split to avoid having data-parallel operations inside branches.
  let (_, _, ret) =
    loop (X1r, betar, retr) = (X1, beta, ret) for r in (r'..<n) do
      let (X1r, betar, recresidr) = loop_body r X1r betar
      let retr[r-k] = recresidr
      in (X1r, betar, retr)

  -- let num_check = r' - k
  -- in (ret, num_check)
  in ret
