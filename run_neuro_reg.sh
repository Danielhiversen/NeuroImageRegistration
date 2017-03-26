#!/bin/bash
###################################################
#
#  Run neruo reg 
#
###################################################
#
#SBATCH -J NeuroRegAllLGG  # sensible name for the job
#SBATCH -p normal            # partition, sinfo to see available
#SBATCH -N 1               # allocate N nodes for the job
#SBATCH -n 40              # n tasks total
#SBATCH --mem=62000        # Memory per node in MegaBytes
#SBATCH --exclusive        # no other jobs on the nodes while job is running
#SBATCH -t 7-00:00         # upper time limit [D-HH:MM] for the job
#SBATCH --output=logs/output_%j.file
#SBATCH --error=logs/error_%j.file
#SBATCH --mail-type=ALL         # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=dahoiv@gmail.com # Email to which notifications will be sent
#
#
#

  source /home/danieli/NeuroImageRegistration/venv2/bin/activate
  python /home/danieli/NeuroImageRegistration/do_img_registration.py
  python /home/danieli/NeuroImageRegistration/do_img_registration_LGG_POST.py
