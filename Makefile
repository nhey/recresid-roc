lib:
	futhark pkg sync

clean:
	rm -f *.out *.out.c *_pyopencl.py
	rm -rf __pycache__

%_pyopencl: %.fut
	futhark pyopencl --library -o $@ $<

%_validate: %_pyopencl
	python $*_validate.py

R_validate:
	nix-shell r-shell.nix --run "python R_validate.py"
