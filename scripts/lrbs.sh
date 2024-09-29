if [[ "$TASK" == "cifar10_vit" ]]; then
    learning_rates=(1e-4 5e-4 1e-3 5e-3)
    batch_sizes=(128 240 480 600)
else
    learning_rates=(5e-4 0.001 0.005 0.01 0.02)
    batch_sizes=(30 60 128 240 480 600)
fi


for batch_size in "${batch_sizes[@]}"; do
for learning_rate in "${learning_rates[@]}"; do
    bash scripts/$TASK.sh --batch-size $batch_size --learning-rate $learning_rate --seed -1 $ARGS_FOR_REPEATED "$@"
done
done
