

iccad%_test: test.py iccad%_config.ini
	python $< $(word 2, $+) 0
