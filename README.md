# ReasonNet


generate data (for data-100m.txt)

```bash
cd data_gen
cargo run --release -- --min 0 --max 6 --size 100000000 --seed 32 --save-file-path data-100m.txt
mv data-100m.txt ../datasets/.
```

for data-50m-add.txt


```bash
cargo run --release -- --min 0 --max 4 --size 5000000 --seed 32 --operators "+" --save-file-path data-50m-add.txt 
```

run the training


```bash
python reason_net/run.py --config-path configs --config-name default.yaml 
```


With small model for test

```bash
python reason_net/run.py --config-path configs --config-name default.yaml module/model=910K
```

generate 

```bash
python reason_net/generate.py lightning_logs/dummy-jv7tadve/last.ckpt 14M "1234+1234=" --interactive
```