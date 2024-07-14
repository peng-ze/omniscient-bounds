for corruption in {0.25,0.50,0.75}; do
    bash scripts/$TASK.sh --label-corrupt-prob $corruption --seed -1 "$@"
done
