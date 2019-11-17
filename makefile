


via_train: train.py via_config.ini
	#python3 write_image_loc_lab.py /research/byu2/hyyang/ICCAD/iccad-official/iccad$*/train ./benchmarks/iccad5/train.txt 1
	mkdir -p ./models/vias
	python3 $< $(word 2, $+) 0

via_test: test.py via_config.ini
	srun --gres=gpu:1 python3 $< $(word 2, $+) 0

via_clean:
	rm -rf ./models/vias/*


via%_20:
	rm -rf dct/20/attack$*/*
	srun -p gpu_24h --gres=gpu:1 python3 dct_attack.py dct_config20$*.ini |& tee dct/20/attack$*/log.txt&

via%_3:
	rm -rf dct/3/attack$*/*
	srun -p gpu_24h --gres=gpu:1 python3 dct_attack.py dct_config03$*.ini |& tee dct/3/attack$*/log.txt&
 
via%_verify: test.py via_config.ini
	mv dct/20/attack$*/log.txt dct/20/attack$*-log.txt
	python3 write_image_loc_lab.py dct/20/attack$*/ dct/attack.txt 1
	srun --gres=gpu:1 python3 $< $(word 2, $+) 0 
