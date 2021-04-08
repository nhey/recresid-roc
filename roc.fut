import "lib/github.com/diku-dk/statistics/statistics"
import "recresid"

module stats = mk_statistics f64

-- Compute the [sample standard
-- deviation](https://en.wikipedia.org/wiki/Standard_deviation),
-- `s = \sqrt{\frac{1}{N-1} \sum_{i=1}^N (x_i - \bar{x})^2}`.
-- let sd [n] (xs: [n]f64) =
--   let nf64 = f64.i64 n
--   let x_mean = (reduce (+) 0 xs)/nf64
--   let sumdiffs = map (\x -> (x - x_mean)**2) xs |> reduce (+) 0
--   in f64.sqrt(sumdiffs/(nf64-1))

-- Empircal fluctuation process containing recursive residuals.
let rcusum [n][k] (bsz: i64) (X: [k][n]f64) (y: [n]f64) =
  let w = recresid bsz (transpose X) y
  let m = length w
  let s = stats.stddev w
  let fr = s * f64.sqrt(f64.i64 (n-k))
  -- TODO: does indexing inhibit performance?
  let standardized = map (\i -> if i == 0 then 0 else w[i-1]/fr) (iota (m+1))
  in scan (+) 0f64 standardized

let pval_brownian_motion_max (x: f64): f64 =
  -- Q is complementary CDF of N(0,1)
  let N = stats.mk_normal {mu=0f64, sigma=1f64}
  let Q = \y -> 1 - stats.cdf N y
  let exp = \y -> f64.e**y
  in 2 * (Q(3*x) + exp(-4*x**2) - exp(-4*x**2) * (Q x))

-- Structural change test for Brownian motion.
let sctest process =
  let xs = process[1:]
  let nf64 = f64.i64 (length xs)
  -- x = max(abs(xs * 1/(1 + 2*j))) where j = 1/n, 2/n, ..., n.
  let div i = 1 + (f64.i64 (2*i+2)) / nf64
  let x = map2 (\x i -> x/(div i) |> f64.abs) xs (indices xs) |> f64.maximum
  in pval_brownian_motion_max x

let boundary n confidence: [n]f64 =
  -- 2*(1 + conf*t) with t in [0,1].
  let div = f64.i64 n - 1
  in map (\i -> confidence + (2*confidence*(f64.i64 i))/div) (iota n)

entry history_roc [n][k] bsz (X: [k][n]f64) (y: [n]f64) level confidence =
  let m = n - k + 1
  let roc = rcusum bsz (map reverse X) y[::-1] :> [m]f64
  let pval = sctest roc
  let bounds = boundary m confidence
  -- index of first time roc crosses the boundary
  let ind = map3 (\i r b -> if f64.abs r > b then i else i64.highest)
                 (indices roc[1:]) roc[1:] bounds[1:]
                 |> reduce_comm i64.min i64.highest
  -- only consider this index if it is statistically signifcant
  let y_start = if ! (f64.isnan pval) && pval < level && ind != i64.highest
                then m - ind - 1
                else 0
  in y_start

-- For the repl,
let level = 0.05f64
let conf = 0.9478989165152716f64
let X = [[1.0f64,86.0f64],[1.0f64,76.0f64],[1.0f64,92.0f64],[1.0f64,90.0f64],[1.0f64,86.0f64],[1.0f64,84.0f64],[1.0f64,93.0f64],[1.0f64,100.0f64],[1.0f64,87.0f64],[1.0f64,86.0f64],[1.0f64,74.0f64],[1.0f64,98.0f64],[1.0f64,97.0f64],[1.0f64,84.0f64],[1.0f64,91.0f64],[1.0f64,34.0f64],[1.0f64,45.0f64],[1.0f64,56.0f64],[1.0f64,44.0f64],[1.0f64,82.0f64],[1.0f64,72.0f64],[1.0f64,55.0f64],[1.0f64,71.0f64],[1.0f64,50.0f64],[1.0f64,23.0f64],[1.0f64,39.0f64],[1.0f64,28.0f64],[1.0f64,32.0f64],[1.0f64,22.0f64],[1.0f64,25.0f64],[1.0f64,29.0f64],[1.0f64,7.0f64],[1.0f64,26.0f64],[1.0f64,19.0f64],[1.0f64,15.0f64],[1.0f64,20.0f64],[1.0f64,26.0f64],[1.0f64,28.0f64],[1.0f64,17.0f64],[1.0f64,22.0f64],[1.0f64,30.0f64],[1.0f64,25.0f64],[1.0f64,20.0f64],[1.0f64,47.0f64],[1.0f64,32.0f64]]
let Xt = transpose X
let y = [62.0f64, 72.0f64, 75.0f64, 55.0f64, 64.0f64, 21.0f64, 64.0f64, 80.0f64, 67.0f64, 72.0f64, 42.0f64, 76.0f64, 76.0f64, 41.0f64, 48.0f64, 76.0f64, 53.0f64, 60.0f64, 42.0f64, 78.0f64, 29.0f64, 48.0f64, 55.0f64, 29.0f64, 21.0f64, 47.0f64, 81.0f64, 36.0f64, 22.0f64, 44.0f64, 15.0f64,  7.0f64, 42.0f64,  9.0f64, 21.0f64, 21.0f64, 16.0f64, 16.0f64,  9.0f64, 14.0f64, 12.0f64, 17.0f64,  7.0f64, 34.0f64,  8.0f64]
entry test = history_roc 1 Xt y level conf
