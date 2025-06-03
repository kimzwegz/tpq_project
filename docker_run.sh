
docker run -it --name tpq \
  -p 9999:9999 \
  -v "$(pwd)":/home/app \
  kimzwegz/pyalgo:latest \
  /bin/bash