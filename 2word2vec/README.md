# 2Word2vec
## Process
1. prepare the pretrain word2vec [GoogleNews-vectors-negative300.bin.gz.](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing) from the official [word2vec website](https://code.google.com/archive/p/word2vec/).
2. Modify the dataset path ( `BASEFILE_DIR` ) in `1dataset_prepare.py`
3. Run `1dataset_prepare.py`
4. Run `2getemb.py` on different datasets

```
python 2getemb.py 0 > ./log/log0.txt 2>&1
python 2getemb.py 1 > ./log/log1.txt 2>&1
python 2getemb.py 2 > ./log/log2.txt 2>&1
python 2getemb.py 3 > ./log/log3.txt 2>&1
python 2getemb.py 4 > ./log/log4.txt 2>&1
```

## Note
`2getemb.py` needs tensorflow 1.15.0. Therefore, you can run `2getemb.py` with docker easily

```bash
# pull docker images
docker pull tensorflow/tensorflow:1.15.0-gpu-py3
# Change the current directory to dpa/.
cd dpa/
# run the docker 
docker run --gpus all  --name code_tf15  -it -v $(pwd):/home tensorflow/tensorflow:1.15.0-gpu-py3
# run these commands inside the docker 
pip install pandas numpy tqdm 
sh run.sh
```


# Acknowledgement.
This code refers code from:        
[Wenhui-Yu/TDAR](https://github.com/Wenhui-Yu/TDAR)