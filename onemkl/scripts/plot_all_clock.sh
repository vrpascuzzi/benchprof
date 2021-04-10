#!/bin/bash
declare -a GENS=("philox" "mrg") ;
declare -a DISTS=("uniform_float" "uniform_double" "gaussian_float" \
    "gaussian_double" "lognormal_float" "bits_int") ;
declare -a APIS=("buffer" "usm") ;
declare -a CLOCKS=("total" "kernel") ;

for gen in ${GENS[@]}; do
    for dist in ${DISTS[@]}; do
        for api in ${APIS[@]}; do
            for clock in ${CLOCKS[@]}; do
                echo "Doing $gen $dist $api $clock"
                python3 python/plot_clock_csv.py $gen $dist $api $clock out/*.csv
            done
        done
    done;
done;