#!/usr/bin/env bash
echo "Starting isolation: 20260904_104034"
echo "Target module: condveg"
echo "Parent subroutine: condveg_main"
echo "Target subroutines: albedo_surface_main"
echo "OpenACC: False, F2PY: True, Tapenade: False", PY2JX: False
echo "=========================================="
fgpt isolate \
    --work "/data/ssivanes" \
    --rest_of_path "modipsl_truck_opt/modeles/ORCHIDEE/src_sechiba/" \
    --target_module "condveg" \
    --parent_subroutine "condveg_main" \
    --target_subroutines albedo_surface_main \
    --openacc "False" \
    --f2py "True" \
    --tapenade "False" \
    --py2jx "False" \
    --config_path "template.yaml" \
    --vectorize kjpindex \
    --mode "jax" \
    --benchmark_dir "benchmark" \

echo "=========================================="
echo "Isolation completed: $(date)"
