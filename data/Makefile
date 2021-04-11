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
	rm -f *.out *.in *.in.py

d-2-10-2.in: gen-data.out
	echo "2 10 2 0.25" | ./gen-data.out > $@

d-2-10-2.in.py: d-2-10-2.in
	cat $< | sed -e "s/f64//g" | sed -e "s/\.nan/float('nan')/g" \
	       | sed -e "0,/\[/{s/\[/X=\[/}" \
	       | sed -e "2,/\[/{s/\[/image=\[/}" > $@
