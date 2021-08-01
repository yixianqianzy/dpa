FROM tensorflow/tensorflow:1.15.0-gpu-py3
SHELL ["/bin/bash", "-c"]
COPY dataset /home/dataset
COPY 2word2vec /home/2word2vec
RUN ls /home 
RUN python /home/2word2vec/2getemb.py 0 \
        && python /home/2word2vec/2getemb.py 1 \
        && python /home/2word2vec/2getemb.py 2 \
        && python /home/2word2vec/2getemb.py 3  \
        && python /home/2word2vec/2getemb.py 4 