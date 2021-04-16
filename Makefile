lib:
	futhark pkg sync

clean:
	rm -f *.out *.out.c *_pyopencl.py
	rm -rf __pycache__


%_pyopencl: %.fut
	futhark pyopencl --library -o $@ $<

objs = recresid_validate roc_validate
$(objs): %_validate: %_pyopencl
	python $@.py

R = R_recresid_validate R_roc_validate
$(R):
	nix-shell r-shell.nix --run "python $@.py"

peru = peru_recresid
$(peru): peru_%: %_pyopencl
	python $*_validate_peru.py

