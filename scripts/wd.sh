for wd in {0,1e-5,1e-4,1e-3}; do
    bash scripts/$TASK.sh  --weight-decay $wd --seed -1 $ARGS_FOR_REPEATED "$@"
done
