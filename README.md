# Intentional_Forgetting
Repository for the Thesis Topic 'Deep Continual Learning with Intentional Forgetting'

## Generating Images
1. ```cd docker```
2. ```docker build -t --rm if_dataset -f Dockerfile .```
3. ```docker run -it -v --rm /localpathto/clevr-hans-dataset-gen:/home/workspace/clevr-hans-dataset-gen --name if_dataset --entrypoint='/bin/bash' --user $(id -u):$(id -g) if_dataset```
4. ```cd ../home/workspace/clevr-hans-dataset-gen/image_generation/```
5. ```./run_scripts/run_conf_3.sh```

To generate positive images, set ``--IF_type == 'True'`` and comment out ``--validation``. Generate negative images with ``--IF_type == 'False'`` and ``--validation == 'False'``. 
To generate positive images for the Doubly Confounded Dataset, ``--IF_type == 'True'`` and ``--validation == 'True'``.

The default output location is ```clevr-hans-dataset-gen/output/``` and is specified in the run shell script.

## Collecting Baselines
1. ```cd clevr-hans-dataset-gen/docker/```
2. ```docker build -t if_train -f Dockerfile .```
3. ```docker run -it --rm  --gpus device=13 -v /localpathto/training_scripts:/workspace/ --name if_train --shm-size=4gb --entrypoint='/bin/bash' if_train```
4. ```./run.sh```

The bash script ```run.sh``` runs all three training schemes as well as generates the task analysis matrices.

## Intentional Forgetting
1. ```cd docker```
2. ```docker build -t --rm if_nesy -f Dockerfile .```
3. ```docker run -it --rm --gpus device=13 -v /localpathto/NeSyXIL-Intentional_Forgetting:/workspace/repositories/NeSyXIL -v /pathto/data/IF_dataset:/workspace/datasets/IF_dataset--name if_nesy --shm-size=4gb --entrypoint='/bin/bash' if_nesy```
4. ```cd workspace/repositories/NeSyXIL/```
5. ```./scripts/clevr-hans-concept-learner_xil.sh 0 0 /workspace/datasets/IF_dataset/```
6. ```./scripts/run_matrices.sh 0 0 /workspace/datasets/IF_dataset/```


