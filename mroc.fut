import "lib/github.com/diku-dk/statistics/statistics"
import "recresid"

module stats = mk_statistics f64

-- Compute the [sample standard
-- deviation](https://en.wikipedia.org/wiki/Standard_deviation),
-- `s = \sqrt{\frac{1}{N-1} \sum_{i=1}^N (x_i - \bar{x})^2}`,
-- but ignoring nans.
let sample_sd_nan [m] (xs: [m]f64) (num_non_nan: i64) =
  let nf64 = f64.i64 num_non_nan
  -- TODO prove associativity
  -- TODO prove commutativity
  let add_nan a b = if f64.isnan a
                    then b
                    else if f64.isnan b
                         then a
                         else a + b
  -- TODO use reduce_comm once proven commutativity
  let x_mean = (reduce add_nan 0 xs)/nf64
  let diffs = map (\x -> if f64.isnan x then 0 else (x - x_mean)**2) xs
  in (f64.sum diffs)/(nf64 - 1) |> f64.sqrt

-- Empircal fluctuation process containing recursive residuals.
-- Outputs recursive CUSUM and number of non nan values _excluding_
-- the prepended zero for each `y` in `ys`.
let rcusum [m][N][k] (X: [N][k]f64) (ys: [m][N]f64) =
  -- TODO use mrecresid_nn and propage Xs_nn ys_nn outwards
  let (w's, _, ns) = mrecresid X ys
  let ws = transpose w's |> trace
  let ns = map (\n -> n - k) ns
  -- compute sample sd, ignoring nans
  let sample_sds = map2 sample_sd_nan ws ns |> trace
  -- Standardize and insert 0 in front.
  let n = N-k+1
  let standardized =
    map3 (\i s num_non_nan ->
           let fr = s * f64.sqrt(f64.i64 num_non_nan)
           in map (\j -> if j == 0 then 0 else ws[i, j-1]/fr) (iota n)
        ) (iota m) sample_sds ns
  in (map (scan (+) 0f64) standardized, ns)

let std_normal_cdf =
  stats.cdf (stats.mk_normal {mu=0f64, sigma=1f64})

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
  -- 2*(1 + conf*t) with t in [0,1].
  let div = f64.i64 n - 1
  in map (\i -> if i < n
                then confidence + (2*confidence*(f64.i64 i))/div
                else f64.nan
         ) (iota N)

entry mhistory_roc [m][N][k] level confidence
                             (X: [N][k]f64) (ys: [m][N]f64): [m](i64, bool) =
  let (rocs, nns) = rcusum (reverse X) (map reverse ys)
  -- TODO fuse pval and bounds and ind (atleast bounds and ind)
  let pvals = map2 sctest rocs nns
  let n = N - k + 1 |> trace
  let bounds = map (boundary confidence n) nns |> trace
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
            -- only consider this index if it is statistically signifcant
            let chk = !(f64.isnan pval) && pval < level && ind != i64.highest 
            let y_start = if chk then nn - ind else 0
            in (y_start, chk)
          ) inds nns pvals

-- For the repl,
let level = 0.05f64
let conf = 0.9478989165152716f64
let X = [[1.0f64,86.0f64],[1.0f64,76.0f64],[1.0f64,92.0f64],[1.0f64,90.0f64],[1.0f64,86.0f64],[1.0f64,84.0f64],[1.0f64,93.0f64],[1.0f64,100.0f64],[1.0f64,87.0f64],[1.0f64,86.0f64],[1.0f64,74.0f64],[1.0f64,98.0f64],[1.0f64,97.0f64],[1.0f64,84.0f64],[1.0f64,91.0f64],[1.0f64,34.0f64],[1.0f64,45.0f64],[1.0f64,56.0f64],[1.0f64,44.0f64],[1.0f64,82.0f64],[1.0f64,72.0f64],[1.0f64,55.0f64],[1.0f64,71.0f64],[1.0f64,50.0f64],[1.0f64,23.0f64],[1.0f64,39.0f64],[1.0f64,28.0f64],[1.0f64,32.0f64],[1.0f64,22.0f64],[1.0f64,25.0f64],[1.0f64,29.0f64],[1.0f64,7.0f64],[1.0f64,26.0f64],[1.0f64,19.0f64],[1.0f64,15.0f64],[1.0f64,20.0f64],[1.0f64,26.0f64],[1.0f64,28.0f64],[1.0f64,17.0f64],[1.0f64,22.0f64],[1.0f64,30.0f64],[1.0f64,25.0f64],[1.0f64,20.0f64],[1.0f64,47.0f64],[1.0f64,32.0f64]]
let Xt = transpose X
let ys = [[62.0f64, 72.0f64, f64.nan, 55.0f64, 64.0f64, f64.nan, 64.0f64, 80.0f64, 67.0f64, 72.0f64, 42.0f64, 76.0f64, 76.0f64, 41.0f64, 48.0f64, 76.0f64, 53.0f64, 60.0f64, 42.0f64, 78.0f64, 29.0f64, 48.0f64, 55.0f64, 29.0f64, 21.0f64, 47.0f64, 81.0f64, 36.0f64, 22.0f64, 44.0f64, 15.0f64,  7.0f64, 42.0f64,  9.0f64, 21.0f64, 21.0f64, 16.0f64, 16.0f64,  9.0f64, 14.0f64, 12.0f64, 17.0f64,  7.0f64, 34.0f64,  8.0f64], [f64.nan, 68.0f64, f64.nan, 55.0f64, 64.0f64, f64.nan, 64.0f64, 80.0f64, 67.0f64, 72.0f64, 42.0f64, 76.0f64, 76.0f64, 41.0f64, 48.0f64, 72.0f64, f64.nan, 60.0f64, 42.0f64, 78.0f64, 29.0f64, 43.0f64, 55.0f64, 29.0f64, 21.0f64, 47.0f64, 81.0f64, 36.0f64, 22.0f64, 44.0f64, 15.0f64,  7.0f64, 40.0f64,  9.0f64, 21.0f64, 21.0f64, 16.0f64, 16.0f64,  9.0f64, 14.0f64, 12.0f64, 17.0f64,  7.0f64, 34.0f64,  8.0f64]]
let test = mhistory_roc level conf X ys
-- let test =
--   let (rocs, nns) = rcusum X ys
--   let pvals = map2 sctest rocs nns
--   let N = length rocs[0]
--   let bounds = map (boundary conf N) nns
--   in bounds
