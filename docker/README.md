# localGPT Docker

## Build Docker
```shell
$ docker build -t local_gpt .
```

## Run Docker
```shell
$ docker run --env MODEL_ID="TheBloke/orca_mini_3B-GGML" --env MODEL_BASENAME="orca-mini-3b.ggmlv3.q4_0.bin" --env TEMP=0 --env N_CTX=2048 --env N_THREADS=2048 -v /<AbsolutePath>/prompt.txt:/root/localGPT/prompt.txt -v /<AbsolutePath>/SOURCE_DOCUMENTS:/root/localGPT/SOURCE_DOCUMENTS -it local_gpt
```