#!/usr/bin/env bash
echo "Starting isolation: 20260822_231502"
echo "Target module: hydrol"
echo "Parent subroutine: hydrol_main"
echo "Target subroutines: hydrol_alma hydrol_vegupd hydrol_canop hydrol_flood hydrol_hydraulic_arch_tuzet_calc hydrol_soil explicitsnow_main"
echo "OpenACC: False, F2PY: False, Tapenade: True", PY2JX: False
echo "=========================================="
fgpt isolate \
    --work "/scratchu/kardaneh" \
    --rest_of_path "tmp/modipsl_truck_opt/modeles/ORCHIDEE/src_sechiba/" \
    --target_module "hydrol" \
    --parent_subroutine "hydrol_main" \
    --target_subroutines hydrol_alma hydrol_vegupd hydrol_canop hydrol_flood hydrol_hydraulic_arch_tuzet_calc hydrol_soil explicitsnow_main \
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
