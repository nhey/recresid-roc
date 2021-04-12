# In all data sets, 25% of the values are NANs.
# Naming legend: d-M-N-n
#   M is number of pixels in image
#   N is the time series length per pixel
#   n is an ensured stable period from the beginning of each time series;
#     the induced break point is some time after n.
gen-data.out: gen-data.fut
	futhark c -o $@ $<
	rm $@.c

clean:
	rm -f *.out *.in

datasets = d-2-50-10.in d-10-100-50.in d-10-200-60.in

$(datasets): %.in: gen-data.out
	echo "$@" | sed -e "s/[d|.in]//g"  \
	          | sed -e "s/^-//" \
	          | sed -e "s/-/ /g" \
	          | sed -e "s/$$/ 0.25/" \
	          | ./gen-data.out > $@
