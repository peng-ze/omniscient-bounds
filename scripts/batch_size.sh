for batch_size in {120,240,480,600}; do
    bash scripts/$TASK.sh --batch-size $batch_size --seed -1 $ARGS_FOR_REPEATED "$@"
done
