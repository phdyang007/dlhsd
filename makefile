

iccad%_test: test.py iccad%_config.ini
	python3 write_image_loc_lab.py /research/byu2/hyyang/ICCAD/iccad-official/iccad$*/test ./benchmarks/iccad5/test.txt 1
	python3 $< $(word 2, $+) 0


iccad%_train: train.py iccad%_config.ini
	python3 write_image_loc_lab.py /research/byu2/hyyang/ICCAD/iccad-official/iccad$*/train ./benchmarks/iccad5/train.txt 1
	python3 $< $(word 2, $+) 0


iccad%_clean:
	rm -rf ./models/iccad$*/*ckpt*