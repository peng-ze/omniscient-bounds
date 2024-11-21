for scaling in {0.25,0.5,0.75,0.9,1.5,2,3}; do
    bash scripts/$TASK.sh --weight-scaling $scaling $ARGS_FOR_REPEATED --seed -1 "$@"
done
