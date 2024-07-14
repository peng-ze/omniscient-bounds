for batch_size in {30,60,120,240,480,600}; do
for learning_rate in {0.001,0.005,0.01,0.02,0.04}; do
    python main.py --arch fc1 --epochs 500 --batch-size $batch_size --learning-rate $learning_rate --dataset mnist --data-path data --width 512 --label-corrupt-prob 0 --early_stop False --fixinit False --seed 1 --k 6 --test-freq 20 "$@"
done
done
