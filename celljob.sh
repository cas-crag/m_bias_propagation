#!/usr/bin/env bash

mkdir /mnt/zfsusers/ccragg/m_bias_prop/oct21_run/base_LCDM/delta_cells_nside_4096

for RANK in {0..20}
do
    addqueue -q blackhole -m 50 -o /mnt/zfsusers/ccragg/m_bias_prop/oct21_run/base_LCDM/delta_cells_nside_4096/cell_nside_4096-%j.out ./namaster_parr.py $RANK
done
