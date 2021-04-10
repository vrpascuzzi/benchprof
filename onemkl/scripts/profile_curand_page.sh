path=prof
prefix=a100_cuda_curand
# for i in 1 10 100 1000 10000 100000 1000000 10000000 100000000 ; do
for i in 1 10 100 10000 100000 1000000 10000000 100000000 ; do
  ofile=${path}/${prefix}.batch_${i}.philox.uniform_float.raw
  echo "doing ${ofile}..."
  srun ncu -o ${ofile} --page details \
  ./bld/test_cuda_curand_philox.exe \
  10 ${i} uniform_float

  cat ${ofile} \
  | grep -E 'generate_seed.*Duration|generate_seed.*SM \[%\]|generate_seed.*Achieved Occupancy' \
    > ${path}/${prefix}.batch_${i}.philox_generate_seed.csv
  cat ${ofile} \
  | grep -E 'gen_sequenced.*Duration|gen_sequenced.*SM \[%\]|gen_sequenced.*Achieved Occupancy' \
    > ${path}/${prefix}.batch_${i}.philox_gen_sequenced.csv
  cat ${ofile} \
  | grep -E 'range_transform.*Duration|range_transform.*SM \[%\]|range_transform.*Achieved Occupancy' \
    > ${path}/${prefix}.batch_${i}.philox_range_transform.csv
done;

prefix=a100_mkl_curand
# for i in 1 10 100 1000 10000 100000 1000000 10000000 100000000 ; do
for i in 1 10 100 10000 100000 1000000 10000000 100000000 ; do
  ofile=${path}/${prefix}.batch_${i}.usm.philox.uniform_float.raw
  echo "doing ${ofile}..."
  srun ncu -o ${ofile} --page details \
  ./bld/test_mkl_rng_curand_usm_philox.exe \
  10 ${i} uniform_float > ${ofile}

  
  cat ${ofile} \
  | grep -E 'generate_seed.*Duration|generate_seed.*SM \[%\]|generate_seed.*Achieved Occupancy' \
    > ${path}/${prefix}.batch_${i}.usm.philox.generate_seed.csv
  cat ${ofile} \
  | grep -E 'gen_sequenced.*Duration|gen_sequenced.*SM \[%\]|gen_sequenced.*Achieved Occupancy' \
    > ${path}/${prefix}.batch_${i}.usm.philox.gen_sequenced.csv
  cat ${ofile} \
  | grep -E 'range_transform.*Duration|range_transform.*SM \[%\]|range_transform.*Achieved Occupancy' \
    > ${path}/${prefix}.batch_${i}.usm.philox.range_transform.csv
done;
for i in 1 10 100 10000 100000 1000000 10000000 100000000 ; do
  ofile=${path}/${prefix}.batch_${i}.buffer.philox.uniform_float.raw
  echo "doing ${ofile}..."
  srun ncu -o ${ofile} --page details \
  ./bld/test_mkl_rng_curand_buffer_philox.exe \
  10 ${i} uniform_float > ${ofile}

  
  cat ${ofile} \
  | grep -E 'generate_seed.*Duration|generate_seed.*SM \[%\]|generate_seed.*Achieved Occupancy' \
    > ${path}/${prefix}.batch_${i}.buffer.philox.generate_seed.csv
  cat ${ofile} \
  | grep -E 'gen_sequenced.*Duration|gen_sequenced.*SM \[%\]|gen_sequenced.*Achieved Occupancy' \
    > ${path}/${prefix}.batch_${i}.buffer.philox.gen_sequenced.csv
  cat ${ofile} \
  | grep -E 'range_transform.*Duration|range_transform.*SM \[%\]|range_transform.*Achieved Occupancy' \
    > ${path}/${prefix}.batch_${i}.buffer.philox.range_transform.csv
done;

# # for i in 1 10 100 1000 10000 100000 1000000 10000000 100000000 ; do
# for i in 1 10 100 10000 100000 1000000 10000000 100000000 ; do
#   ofile=${path}/${prefix}.100_${i}.usm.philox.uniform_double.raw
#   echo "doing ${ofile}..."
#   /opt/nvidia/nsight-compute/2020.3/ncu --csv --page details \
#   ./exe/test_mkl_rng_curand_usm_philox.exe \
#   1 ${i} uniform_double > ${ofile}

  
#   cat ${ofile} \
#   | grep -E 'generate_seed.*Duration|generate_seed.*SM \[%\]|generate_seed.*Achieved Occupancy' \
#     > ${path}/${prefix}.100_${i}.usm.philox.generate_seed.csv
#   cat ${ofile} \
#   | grep -E 'gen_sequenced.*Duration|gen_sequenced.*SM \[%\]|gen_sequenced.*Achieved Occupancy' \
#     > ${path}/${prefix}.100_${i}.usm.philox.gen_sequenced.csv
#   cat ${ofile} \
#   | grep -E 'range_transform.*Duration|range_transform.*SM \[%\]|range_transform.*Achieved Occupancy' \
#     > ${path}/${prefix}.100_${i}.usm.philox.range_transform.csv
# done;
