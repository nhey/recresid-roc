lib:
	futhark pkg sync

clean:
	rm -f *.out *.out.c *_pyopencl.py
	rm -rf __pycache__


pyopencl_libs=recresid_pyopencl.py roc_pyopencl.py mroc_pyopencl.py
$(pyopencl_libs): %_pyopencl.py: %.fut
	futhark pyopencl --library -o $*_pyopencl $<

objs = recresid_validate roc_validate
$(objs): %_validate: %_pyopencl.py
	python $@.py

R = R_recresid_validate R_roc_validate
$(R):
	nix-shell r-shell.nix --run "python $@.py"

data_recresid = rand_recresid realworld_recresid
$(data_recresid): %_recresid: recresid_pyopencl.py
	python recresid_validate_$*.py

data_roc = rand_roc realworld_roc
$(data_roc): %_roc: mroc_pyopencl.py
	python roc_validate_$*.py
