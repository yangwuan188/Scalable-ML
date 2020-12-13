#!/bin/bash
#$ -l h_rt=4:00:00  #time needed
#$ -pe smp 20 #number of cores
#$ -l rmem=3G #number of memery
#$ -P rse-com6012
#$ -q rse-com6012.q
#$ -o Q2-20cores.output #This is where your output and errors are logged.
#$ -j y # normal and error outputs into a single file (the file above)
#$ -M wyang36@sheffield.ac.uk #Notify you by email, remove this line if you don't like
#$ -m ea #Email you when it finished or aborted
#$ -cwd # Run job from current directory

module load apps/java/jdk1.8.0_102/binary

module load apps/python/conda

source activate myspark

spark-submit ./Code/Q2.py
