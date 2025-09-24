#!/bin/bash

source ../../.env
source "$SEQ2LOC_ENV/bin/activate"

python ../../main.py --sweep_config ../../configs/config_esm2.yaml
python ../../main.py --sweep_config ../../configs/config_esm3.yaml
python ../../main.py --sweep_config ../../configs/config_prott5.yaml
python ../../main.py --sweep_config ../../configs/config_protbert.yaml