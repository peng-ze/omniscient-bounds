for width in {0.1,0.25,0.5,0.75,1.25,1.5,2.0}; do
    bash scripts/$TASK.sh  --width $width --seed -1 $ARGS_FOR_REPEATED "$@"
done
