#!/bin/bash
#
#SBATCH --job-name=test
#SBATCH --output=res.txt
#SBATCH --gres=gpu:1
#SBATCH --time=60:00

touch output.txt
echo "Hello World" >> output.txt
