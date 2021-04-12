lib:
	futhark pkg sync

clean:
	rm -f *.out *.out.c *_pyopencl.py
	rm -rf __pycache__

objs = recresid_validate roc_validate

%_pyopencl: %.fut
	futhark pyopencl --library -o $@ $<

$(objs): %_validate: %_pyopencl
	python $@.py

R_validate:
	nix-shell r-shell.nix --run "python R_validate.py"
