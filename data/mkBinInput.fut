import "gen-data"
entry main [m][N] (trend: i32) (k: i32) (n: i32) (freq: f32)
                  (hfrac: f32) (lam: f32)
                  (mappingindices : [N]i32)
                  (image : [m][N]f32) =
  let X = mkX_with_trend (i64.i32 k*2+2) (f64.f32 freq)
                         (map i64.i32 mappingindices)
  in (X, map (map f64.f32) image)
