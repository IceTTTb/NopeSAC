## 1. Build the dockerfile
```bash
docker build - < <path-to-nopesac>/docker/Dockerfile --tag nopesac:latest
```

## 2. Run the docker image
```bash
docker run -it --entrypoint bash --name nopesac --gpus all nopesac:latest
```
