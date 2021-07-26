# Evaluation


## How to run
```bash
python maml.py --task sensitive --supportdata ele+movies+musics --gpu 1 --num_epoch 30 --dataset cd --savename cd_ele_movies_musics --g_name Sen_mi5 
```
You can modify the detailed parameters according to the definition in main.py.


Some important parameters are listed as following:

- `supportdata`: source domain adaptation dataset.
- `dataset`: target domain dataset.
- `g_name`: should be same with the parameters of `method_name` from `../3generation/main.py` 

# Acknowledgement.
This code refers code from:
[hoyeoplee/MeLU](https://github.com/hoyeoplee/MeLU).

