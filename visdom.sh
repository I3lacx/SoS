#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate sos
python -m visdom.server
