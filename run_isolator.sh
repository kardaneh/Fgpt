#!/usr/bin/env bash
echo "Starting isolation: 20260810_101522"
echo "Target module: condveg"
echo "Parent subroutine: condveg_main"
echo "Target subroutines: albedo_surface_main"
echo "OpenACC: False, F2PY: False, Tapenade: True", PY2JX: False
echo "=========================================="
fgpt isolate \
    --work "/scratchu/kardaneh" \
    --rest_of_path "tmp/modipsl_truck_opt/modeles/ORCHIDEE/src_sechiba/" \
    --target_module "condveg" \
    --parent_subroutine "condveg_main" \
    --target_subroutines albedo_surface_main \
    --openacc "False" \
    --f2py "False" \
    --tapenade "True" \
    --py2jx "False" \
    --config_path "template.yaml" \
    --vectorize kjpindex \
    --mode "jax" \
    --benchmark_dir "benchmark" \

echo "=========================================="
echo "Isolation completed: $(date)"
