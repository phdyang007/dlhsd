


via_train: train.py via_config.ini
	#python3 write_image_loc_lab.py /research/byu2/hyyang/ICCAD/iccad-official/iccad$*/train ./benchmarks/iccad5/train.txt 1
	mkdir -p ./models/vias
	python3 $< $(word 2, $+) 0

via_test: test.py via_config.ini
	python3 $< $(word 2, $+) 0

via_clean:
	rm -rf ./models/vias/*

