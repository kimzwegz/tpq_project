docker build -t miniconda --no-cache .
docker run -it -h miniconda -p 9999:9999 -v "$(pwd)":/home/dev --name pyalgo_minic miniconda /bin/bash