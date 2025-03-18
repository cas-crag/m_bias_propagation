#!/usr/bin/env bash                                                                                                                                             
for RANK in {0..4}
do
    for BIAS_L in 50 500 1000 1500 2000 2500 3000 3500 4000 4500 5000
    do
	addqueue -q blackhole -m 50 -o /mnt/zfsusers/ccragg/m_bias_prop/rms/base_LCDM/bias_nside_4096_rms_${RANK}/bias_nside_4096_b50-%j.out ./nam_bias_parr.py $RANK $BIAS_L
    done
done
