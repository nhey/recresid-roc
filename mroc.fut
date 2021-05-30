import "lib/github.com/diku-dk/statistics/statistics"
import "recresid"

module stats = mk_statistics f64

-- Compute the [sample standard
-- deviation](https://en.wikipedia.org/wiki/Standard_deviation),
-- `s = \sqrt{\frac{1}{N-1} \sum_{i=1}^N (x_i - \bar{x})^2}`,
-- but ignoring nans.
let sample_sd_nan [m] (xs: [m]f64) (num_non_nan: i64) =
  let nf64 = f64.i64 num_non_nan
  -- Associativity proof.
  -- Let `+` be add_nan. For this to be associative,
  --   (a + b) + c = a + (b + c)
  -- must hold. Each operand can either be nan or not.
  -- So we get the following truth table:
  -- isnan a | isnan b | isnan c | LHS                 | RHS
  --       T |       T |       T | a + c = c = nan     | a + b = b = nan
  --       F |       T |       T | a + c = a           | a + b = a
  --       F |       F |       T | a + b               | a + b
  --       F |       F |       F | a + b + c           | a + b + c
  --       T |       F |       F | b + c               | b + c
  --       T |       T |       F | b + c = nan + c = c | a + c = nan + c = c
  --       T |       F |       T | b + c = b + nan = b | a + b = nan + b = b
  --       F |       T |       F | a + c               | a + c
  -- Of course this is a lie because we operate on floats.
  let add_nan a b = if f64.isnan a
                    then b
                    else if f64.isnan b
                         then a
                         else a + b
  -- TODO prove commutativity and use `reduce_comm` (probably no speedup)
  let x_mean = (reduce add_nan 0 xs)/nf64
  let diffs = map (\x -> if f64.isnan x then 0 else (x - x_mean)**2) xs
  in (f64.sum diffs)/(nf64 - 1) |> f64.sqrt

-- Empircal fluctuation process containing recursive residuals.
-- Outputs recursive CUSUM and number of non nan values _excluding_
-- the prepended zero for each `y` in `ys`.
let rcusum [m][N][k] (X: [N][k]f64) (ys: [m][N]f64) =
  let (wTs, _, ns) = mrecresid X ys
  let ws = transpose wTs
  -- Standardize and insert 0 in front.
  let Nmk = N-k+1
  let (process, ns) = unzip <|
    map2 (\w npk ->
           let n = npk - k -- num non nan recursive residuals
           let s = sample_sd_nan w n
           let fr = s * f64.sqrt(f64.i64 n)
           let sdized = map (\j -> if j == 0 then 0 else w[j-1]/fr) (iota Nmk)
           in (scan(+) 0f64 sdized, n)
        ) ws ns
  in (process, ns)

let std_normal_cdf =
  stats.cdf (stats.mk_normal {mu=0f64, sigma=1f64})

-- TODO this may be done cheaper for x < 0.3; see
--      R strucchange `pvalue.efp`.
--      Also not saving intermediate results.
let pval_brownian_motion_max (x: f64): f64 =
  -- Q is complementary CDF of N(0,1)
  let Q = \y -> 1 - std_normal_cdf y
  let exp = \y -> f64.e**y
  in 2 * (Q(3*x) + exp(-4*x**2) - exp(-4*x**2) * (Q x))

-- Structural change test for Brownian motion.
-- `num_non_nan` is without the initial zero in process.
let sctest [n] (process: [n]f64) (num_non_nan: i64) : f64 =
  let nf64 = f64.i64 num_non_nan
  let xs = process[1:]
  -- x = max(abs(xs * 1/(1 + 2*j))) where j = 1/n, 2/n, ..., n.
  let div i = 1 + (f64.i64 (2*i+2)) / nf64
  let x = f64.maximum <| map2 (\x i -> if f64.isnan x
                                       then -f64.inf
                                       else f64.abs (x/(div i))
                              ) xs (indices xs)
  in pval_brownian_motion_max x

-- `N` is padded length.
-- `nm1` is number of non-nan values excluding inital zero.
let boundary confidence N nm1: [N]f64 =
  let n = nm1 + 1
  -- conf*(1 + 2*t) with t in [0,1].
  let div = f64.i64 n - 1
  in map (\i -> if i < n
                then confidence + (2*confidence*(f64.i64 i))/div
                else f64.nan
         ) (iota N)

-- TODO handle all nan input
-- TODO fuse maps around inner sizes
-- Map distributed stable history computation.
-- entry mhistory_roc [m][N][k] level confidence
--                              (X: [N][k]f64) (ys: [m][N]f64) =
--   let (rocs, nns) = rcusum (reverse X) (map reverse ys)
--   -- TODO fuse pval and bounds and ind, if same inner sizes
--   let pvals = map2 sctest rocs nns
--   let n = N - k + 1
--   let bounds = map (boundary confidence n) nns
--   -- index of first time roc crosses the boundary
--   let inds =
--     map2 (\roc bound ->
--             let nm1 = n - 1
--             let roc = roc[1:] :> [nm1]f64
--             let bound = bound[1:] :> [nm1]f64
--             in map3 (\i r b ->
--                        if f64.abs r > b
--                        then i
--                        else i64.highest
--                     ) (iota nm1) roc bound
--                |> reduce_comm i64.min i64.highest
--          ) rocs bounds
--   in map3 (\ind nn pval ->
--             let chk = !(f64.isnan pval) && pval < level && ind != i64.highest
--             let y_start = if chk then nn - ind else 0
--             in y_start
--           ) inds nns pvals

entry mhistory_roc [m][N][k] level confidence
                             (X: [N][k]f64) (ys: [m][N]f64) =
  -- COMPUTE ROC
  -- Empircal fluctuation process containing recursive residuals.
  -- Outputs recursive CUSUM and number of non nan values _excluding_
  -- the prepended zero for each `y` in `ys`.
  let (wTs, _, ns) = mrecresid (reverse X) (map reverse ys)
  let ws = transpose wTs
  -- Standardize and insert 0 in front.
  let Nmk = N-k+1
  let (rocs, nns, pvals) = unzip3 <|
    map2 (\w npk ->
           let n = npk - k -- num non nan recursive residuals
           let s = sample_sd_nan w n
           let fr = s * f64.sqrt(f64.i64 n)
           let sdized = map (\j -> if j == 0 then 0 else w[j-1]/fr) (iota Nmk)
           let roc = scan (+) 0f64 sdized
           -- Structural change test for Brownian motion.
           -- `num_non_nan` is without the initial zero in process.
           let nf64 = f64.i64 n
           let xs = roc[1:]
           let div i = 1 + (f64.i64 (2*i+2)) / nf64
           let x = f64.maximum <| map2 (\x i -> f64.abs (x/(div i))) xs (indices xs)
           let pval = pval_brownian_motion_max x
           in (roc, n, pval)
        ) ws ns
  let n = N - k + 1
  let bounds = map (boundary confidence n) nns
  -- index of first time roc crosses the boundary
  let inds =
    map2 (\roc bound ->
            let nm1 = n - 1
            let roc = roc[1:] :> [nm1]f64
            let bound = bound[1:] :> [nm1]f64
            in map3 (\i r b ->
                       if f64.abs r > b
                       then i
                       else i64.highest
                    ) (iota nm1) roc bound
               |> reduce_comm i64.min i64.highest
         ) rocs bounds
  in map3 (\ind nn pval ->
            let chk = !(f64.isnan pval) && pval < level && ind != i64.highest
            let y_start = if chk then nn - ind else 0
            in y_start
          ) inds nns pvals
