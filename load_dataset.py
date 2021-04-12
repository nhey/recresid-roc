import numpy as np

def load_fut_data(filename, np_dtype=np.float64, f_dtype="f64"):
  parse = lambda line: line.replace(f_dtype,"").replace(".nan", "float('nan')")
  with open(filename, "r") as f:
    Xt = np.array(eval(parse(f.readline())), dtype=np_dtype)
    image = np.array(eval(parse(f.readline())), dtype=np_dtype)
  return (Xt, image)

