for batch_size in {30,60,120,240,480,600}; do
for learning_rate in {1e-4,5e-4,0.001,0.005,0.01,0.02,0.04}; do
    bash scripts/$TASK.sh --batch-size $batch_size --learning-rate $learning_rate --seed -1 $ARGS_FOR_REPEATED "$@"
done
done
