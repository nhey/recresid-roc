# In all data sets, 25% of the values are NANs.
# Naming legend: d-M-N-n
#   M is number of pixels in image
#   N is the time series length per pixel
#   n is an ensured stable period from the beginning of each time series;
#     the induced break point is some time after n.

# gen-data.fut is adapted from github.com/diku-dk/futhark-kdd19
gen-data.out: gen-data.fut
	futhark c -o $@ $<
	rm $@.c

mkBinInput.out: mkBinInput.fut
	futhark c -o $@ $<
	rm $@.c

clean:
	rm -f *.out *.in

datasets = d-2-20-10.in d-10-100-50.in d-10-200-60.in d-50-500-100.in d-100-100-50.in d-5000-100-50.in

$(datasets): %.in: gen-data.out
	echo "$@" | sed -e "s/[d|.in]//g"  \
	          | sed -e "s/^-//" \
	          | sed -e "s/-/ /g" \
	          | sed -e "s/$$/ 0.25/" \
	          | ./gen-data.out > $@

realdata = real/sahara.in real/peru.in real/africa.in
$(realdata): real/%.in: mkBinInput.out
	mkdir -p real
	gzip -d --keep $*/$*.in.gz
	./mkBinInput.out --binary-output < $*/$*.in > $@
	rm $*/*.in

# N=67968, k2p2 = 8, freq = 12
sahara/X.in: gen-data.out
	echo "67968 8 12" | ./gen-data.out --entry=mkX_simple > $@

sahara/sahara.in: mkBinInput.out
	gzip -dk sahara/sahara.in.gz
	./mkBinInput.out --binary-output < sahara/sahara.in > $@
