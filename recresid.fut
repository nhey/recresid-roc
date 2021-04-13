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
          let model = ols.fit bsz X[:r+1, :] y[:r+1]
          let nona = nonans(betar) && nonans(model.params)
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
  in ret


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

-- Map-distributed recresid.
entry mrecresid [m][N][k] (bsz: i64) (X: [N][k]f64) (ys: [m][N]f64) =
  let tol = f64.sqrt(f64_eps) / (f64.i64 k)

  -- Rearrange `ys` so that valid values come before nans.
  let (ns, ys_nn, indss_nn) = unzip3 (map filter_nan_pad ys)

  -- Initialise recursion.
  let padding = replicate k f64.nan
  let Xs_nn: *[m][N][k]f64 =
    map (\inds_nn ->
           map (\i -> if i != -1 then X[i, :] else padding) inds_nn
        ) indss_nn |> trace

  let _sanity_check = map (\n -> assert (n > k) true) ns

  -- TODO: Fuse with Xs_nn loop above?
  let (X1s, betas) = map2 (\X_nn y_nn ->
                             let model = ols.fit bsz X_nn[:k, :] y_nn[:k]
                             in (model.cov_params, model.params)
                          ) Xs_nn ys_nn |> unzip

  let num_recresids_padded = N - k
  let rets = replicate (m*num_recresids_padded) 0
             |> unflatten num_recresids_padded m

  -- Map is interchanged so that it is inside the sequential loop.
  let (_, r', X1s, betas, retsT) =
    loop (check, r, X1rs, betars, retrs) = (true, k, X1s, betas, rets)
         while check && r < N do
      -- let xs = map (\X_nn -> X_nn[r, :]) Xs_nn
      -- let ds = map (\X1r -> mvmul_filt x X1r x)
      -- let frs = map2 (\x d -> 1 + (dotprod_nan x d)) xs ds
      -- let recresidrs = map4 (\y xy[r] - dotprod_nan x betar
      let (checks, X1rs, betars, recresidrs) = unzip4 <|
        map4 (\X1r betar X_nn y_nn ->
                -- Compute recursive residual
                let x = X_nn[r, :]
                let d = mvmul_filt x X1r x
                let fr = 1 + (dotprod_nan x d)
                let resid = y_nn[r] - dotprod_nan x betar
                let recresidr = resid / f64.sqrt(fr)

                -- Update formulas
                let ddT = linalg.outer d d -- TODO nans in this?
                -- X1r = X1r - ddT/fr
                let X1r = map2 (map2 (\x y -> x - y/fr)) X1r ddT
                -- beta = beta + X1 x * resid
                let betar = map2 (+) betar (map (dotprod_nan x >-> (*resid)) X1r)

                -- Check numerical stability (rectify if unstable)
                let (check, X1r, betar) =
                  if check && (r+1 < N) then
                    -- We check update formula value against full OLS fit
                    let rp1 = r+1
                    let model = ols.fit bsz X_nn[:rp1, :] y_nn[:rp1]
                    let nona = nonans(betar) && nonans(model.params)
                    let allclose = map2 (-) model.params betar
                                   |> all (\x -> f64.abs x <= tol)
                    in (!(nona && allclose), model.cov_params, nan_to_num 0 model.params)
                 else (check, X1r, betar)
                in (check, X1r, betar, recresidr)
             ) X1rs betars Xs_nn ys_nn
      let _show = recresidrs |> trace
      let _loop = (r, N-k) |> trace
      -- let retrs[r-k, :] = recresidrs
      -- in (reduce_comm (||) false checks, r+1, X1rs, betars, retrs)
      in (reduce_comm (||) false checks, r+1, X1rs, betars, rets)

  -- let (_, _, rets) =
  --   loop (X1rs, betars, retrs) = (X1s, betas, rets) for r in (r'..<N) do
  --     -- let xs = map (\X_nn -> X_nn[r, :]) Xs_nn
  --     -- let ds = map (\X1r -> mvmul_filt x X1r x)
  --     -- let frs = map2 (\x d -> 1 + (dotprod_nan x d)) xs ds
  --     -- let recresidrs = map4 (\y xy[r] - dotprod_nan x betar
  --     let (X1rs, betars, recresidrs) = unzip3 <|
  --       map4 (\X1r betar X_nn y_nn ->
  --               -- Compute recursive residual
  --               let x = X_nn[r, :]
  --               let d = mvmul_filt x X1r x
  --               let fr = 1 + (dotprod_nan x d)
  --               let resid = y_nn[r] - dotprod_nan x betar
  --               let recresidr = resid / f64.sqrt(fr)

  --               -- Update formulas
  --               let ddT = linalg.outer d d -- TODO nans in this?
  --               -- X1r = X1r - ddT/fr
  --               let X1r = map2 (map2 (\x y -> x - y/fr)) X1r ddT
  --               -- beta = beta + X1 x * resid
  --               let betar = map2 (+) betar (map (dotprod_nan x >-> (*resid)) X1r)
  --               in (X1r, betar, recresidr)
  --            ) X1rs betars Xs_nn ys_nn
  --     let retrs[:, r-k] = recresidrs
  --     in (X1rs, betars, retrs)

  -- NOTE: this could maybe be replaced by an if-statement in the loop,
  -- which is probably desirable if the loop body is fully sequentialized.
  -- let _sz = assert (n' - k > 0) 0
  in retsT










--- For the repl,
let X' = [[1.000000f64, 1.000000f64, 1.000000f64, 1.000000f64, 1.000000f64, 1.000000f64, 1.000000f64, 1.000000f64, 1.000000f64, 1.000000f64, 1.000000f64, 1.000000f64, 1.000000f64, 1.000000f64, 1.000000f64, 1.000000f64, 1.000000f64, 1.000000f64, 1.000000f64, 1.000000f64], [1.000000f64, 2.000000f64, 3.000000f64, 4.000000f64, 5.000000f64, 6.000000f64, 7.000000f64, 8.000000f64, 9.000000f64, 10.000000f64, 11.000000f64, 12.000000f64, 13.000000f64, 14.000000f64, 15.000000f64, 16.000000f64, 17.000000f64, 18.000000f64, 19.000000f64, 20.000000f64], [0.500000f64, 0.866025f64, 1.000000f64, 0.866025f64, 0.500000f64, 0.000000f64, -0.500000f64, -0.866025f64, -1.000000f64, -0.866025f64, -0.500000f64, -0.000000f64, 0.500000f64, 0.866025f64, 1.000000f64, 0.866025f64, 0.500000f64, 0.000000f64, -0.500000f64, -0.866025f64], [0.866025f64, 0.500000f64, 0.000000f64, -0.500000f64, -0.866025f64, -1.000000f64, -0.866025f64, -0.500000f64, -0.000000f64, 0.500000f64, 0.866025f64, 1.000000f64, 0.866025f64, 0.500000f64, 0.000000f64, -0.500000f64, -0.866025f64, -1.000000f64, -0.866025f64, -0.500000f64], [0.866025f64, 0.866025f64, 0.000000f64, -0.866025f64, -0.866025f64, -0.000000f64, 0.866025f64, 0.866025f64, 0.000000f64, -0.866025f64, -0.866025f64, -0.000000f64, 0.866025f64, 0.866025f64, 0.000000f64, -0.866025f64, -0.866025f64, -0.000000f64, 0.866025f64, 0.866025f64], [0.500000f64, -0.500000f64, -1.000000f64, -0.500000f64, 0.500000f64, 1.000000f64, 0.500000f64, -0.500000f64, -1.000000f64, -0.500000f64, 0.500000f64, 1.000000f64, 0.500000f64, -0.500000f64, -1.000000f64, -0.500000f64, 0.500000f64, 1.000000f64, 0.500000f64, -0.500000f64], [1.000000f64, 0.000000f64, -1.000000f64, -0.000000f64, 1.000000f64, 0.000000f64, -1.000000f64, -0.000000f64, 1.000000f64, 0.000000f64, -1.000000f64, -0.000000f64, 1.000000f64, 0.000000f64, -1.000000f64, -0.000000f64, 1.000000f64, 0.000000f64, -1.000000f64, -0.000000f64], [0.000000f64, -1.000000f64, -0.000000f64, 1.000000f64, 0.000000f64, -1.000000f64, -0.000000f64, 1.000000f64, 0.000000f64, -1.000000f64, 0.000000f64, 1.000000f64, 0.000000f64, -1.000000f64, 0.000000f64, 1.000000f64, 0.000000f64, -1.000000f64, -0.000000f64, 1.000000f64]]
let ys = [[4523.875861f64, f64.nan, 4011.662068f64, 6939.676970f64, 5146.940712f64, 7975.103436f64, 6217.871075f64, 6854.595361f64, 6172.594314f64, 5300.062474f64, 7315.675860f64, 6989.361224f64, f64.nan, 4319.484241f64, 823.677255f64, 4724.745944f64, 3211.334831f64, f64.nan, 4343.563695f64, 3163.030112f64], [6142.967300f64, 7174.472021f64, 6938.832836f64, f64.nan, 4399.747538f64, 4213.380201f64, 4075.672258f64, 4775.551034f64, 4623.968859f64, 7600.757643f64, 4172.123361f64, 4566.777123f64, f64.nan, 6898.511013f64, 6025.019390f64, f64.nan, 5710.909328f64, 4130.184217f64, 3122.243613f64, 3821.353439f64]]

let X = transpose X'
let test = mrecresid 1i64 X ys

-- let res = map filter_nan_pad ys
-- let ns = map (.0) res
-- let ys_nn = map (.1) res
-- let indss_nn = map (.2) res
-- let Xs_nn =
--   map (\inds_nn ->
--          map (\i -> if i != -1 then copy X[i, :] else replicate 8 f64.nan) inds_nn
--       ) indss_nn |> trace
-- let n = ns[0]
-- let y_nn = ys_nn[0,:n]
-- let X_nn = Xs_nn[0,:n]
