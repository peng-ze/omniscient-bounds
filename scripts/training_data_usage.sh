for usage in {0.1,0.25,0.5,0.75}; do
    bash scripts/$TASK.sh --training-data-usage $usage $ARGS_FOR_REPEATED --seed -1 "$@"
done
