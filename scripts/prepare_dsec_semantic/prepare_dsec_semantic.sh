#!/bin/bash

input_path_train=/path/to/DSEC/DSEC_train
input_path_test=/path/to/DSEC/DSEC_test
output_path=../../data/DSEC

seqs=(zurich_city_00_a zurich_city_01_a zurich_city_02_a zurich_city_04_a zurich_city_05_a zurich_city_06_a zurich_city_07_a zurich_city_08_a)

for seq in ${seqs[*]}
do
    python process_dsec_semantic.py --seqpath $input_path_train/$seq --outpath $output_path/$seq
done

seqs=(zurich_city_13_a zurich_city_14_c zurich_city_15_a)
for seq in ${seqs[*]}
do
    python process_dsec_semantic.py --seqpath $input_path_test/$seq --outpath $output_path/$seq
done

