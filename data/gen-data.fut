-- | Program for generating various "random" datasets
--   in which the values for "(M, N, n, nanfreq)" are
--   given as input, where "M" denotes the number of pixels,
--   "N" denotes the timeseries length, "n" denotes the
--   length of the training set, and "nanfreq" denotes the
--   frequency of NAN values in the image. 

import "lib/github.com/diku-dk/cpprandom/random"

module distf64 = uniform_real_distribution f64 minstd_rand
module disti64 = uniform_int_distribution  i64 minstd_rand

let genRands (q: i64) : [q]f64 =
  let arr = replicate q 0.0
  let rng = minstd_rand.rng_from_seed [123] in
  let (arr, _) = 
    loop (arr,rng) for i < q do
        let (rng, x) = distf64.rand (1,6) rng
        let arr[i] = x
        in  (arr, rng)
  in arr

let mkX_with_trend [N] (k2p2: i64) (f: f64) (mappingindices: [N]i64): [k2p2][N]f64 =
  map (\ i ->
        map (\ind ->
                if i == 0 then 1f64
                else if i == 1 then f64.i64 ind
                else let (i', j') = (f64.i64 (i / 2), f64.i64 ind)
                     let angle = 2f64 * f64.pi * i' * j' / f
                     in  if i % 2 == 0 then f64.sin angle
                                       else f64.cos angle
            ) mappingindices
      ) (iota k2p2)

-- for example, something similar to sahara dataset can be generated
-- with the arguments:
-- 67968i64 414i64 1i64 3i64 228i64 12.0f64 0.25f64 1.736126f64 0.5f64
let main (M: i64) (N: i64) (n: i64) (nanfreq: f64) :
         ([][N]f64, [M][N]f64) =
  -- let trend = 1i64
  let k     = 3i64
  let freq  = 12f64 -- for peru, 365f64 for sahara
  -- let hfrac = 0.25f64
  -- let lam   = 1.736126f64

  -- for simplicity take the mapping indices from 1..N
  let mappingindices = map (+1) (iota N)
  let rngi = minstd_rand.rng_from_seed [246]

  -- initialize the image
  let image = replicate M (replicate N f64.nan)
  let (image, _) =
    loop (image, rngi) for i < M do
        -- init the floating-point seed.
        let rngf     = minstd_rand.rng_from_seed [123+(i32.i64 i)]
        let rngf_nan = minstd_rand.rng_from_seed [369+(i32.i64 i)]
        -- compute the break point.
        let (rngi, b0) = disti64.rand (1, N-n-1) rngi
        let break = b0 + n
        -- fill in the time-series up to the breaking point with
        -- values in interval (4000, 8000) describing raining forests.
        let (image, rngf, rngf_nan) =
            loop (image, rngf, rngf_nan) for j < break do
                let (rngf_nan, q) = distf64.rand (0, 1) rngf_nan in
                if q < nanfreq then (image, rngf, rngf_nan)
                else let (rngf, x) = distf64.rand (4000, 8000) rngf
                     let image[i,j] = x
                     in  (image, rngf, rngf_nan)
        -- fill in the points after the break.
        let (image, _rngf, _rngf_nan) =
            loop (image, rngf, rngf_nan) for j0 < N-break do
                let (rngf_nan, q) = distf64.rand (0, 1) rngf_nan in
                if q < nanfreq then (image, rngf, rngf_nan)
                else let j = j0 + break
                     let (rngf, x) = distf64.rand (0, 5000) rngf
                     let image[i,j] = x
                     in  (image, rngf, rngf_nan)
        in  (image, rngi)
  -- in (trend, k, n, freq, hfrac, lam, mappingindices, image)
  let k2p2 = 2 * k + 2
  let X = mkX_with_trend k2p2 freq mappingindices
  in (X, image)

