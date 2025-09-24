#!/bin/bash

# Activate environment
source ../../.env
source "$SEQ2LOC_ENV/bin/activate"

srun python ../../main_ppi.py --sweep_config ../../configs/config_esm2_ppi.yaml
srun python ../../main_ppi.py --sweep_config ../../configs/config_prott5_ppi.yaml