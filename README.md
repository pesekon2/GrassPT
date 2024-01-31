# GrassPT
GPT helper for GRASS GIS

## Usage

Build the docker file (it is recommended to run it with nohup as the training takes some time):

```
nohup docker build -t grasspt:1.0 -f docker/Dockerfile . &
```

Ask a query:

```
docker run --rm --gpus all grasspt:1.0 python3 /src/query.py " How to get metadata of a raster map in GRASS GIS?"
```
