if [[ "$TASK" == "cifar10_vit" ]]; then
    learning_rates=(1e-4 5e-4 1e-3 2e-3 4e-3 8e-3 1e-2 2e-2)
    batch_sizes=(480 240 120 60 30)
else
    learning_rates=(5e-4 0.001 0.005 0.01 0.02 0.1)
    batch_sizes=(60 480 240 120 30)
fi


for batch_size in "${batch_sizes[@]}"; do
for learning_rate in "${learning_rates[@]}"; do
    bash scripts/$TASK.sh --batch-size $batch_size --learning-rate $learning_rate --seed -1 $ARGS_FOR_REPEATED "$@"
done
done
