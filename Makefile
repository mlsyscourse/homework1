FILES= *.py  \
	   *.txt \
	   tests/test_customized_cases.py

handin.tar: $(FILES)
	tar cvf handin.tar $(FILES)

clean:
	rm -f *~ handin.tar
