# DLHSD

## Dependencies

numpy, tensorflow (tested on 1.3 and 1.9), pandas, json, ConfigParser, progress

## Test

e.g. to test iccad1 of dac17, you need to modify iccad1\_config.ini

set ```model_path=./models/iccad1/bl/model.ckpt```

set ```aug=0``` and

 ```make iccad1_test``` in ```dlhsd``` directory

 ## Train

 e.g. to train iccad1 of dac17, you need to modify iccad1\_config.ini

set ```save_path=./models/iccad1/bl/model.ckpt```

set ```aug=0``` and 

```python train_dac.py iccad1_config.ini <gpu_id>```

use ```train.py``` if you want to see some results of the TCAD extension

## Updates

-20180705: Fix a bug that learning rate does not decay properly which might cause unstable results.
