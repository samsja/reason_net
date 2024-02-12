# ReasonNet


generate data (for data-100m.txt)

```bash
cd data_gen
cargo run --release -- --min 3 --max 5 --size 25000000 --seed 32 --save-file-path data-25m-add.txt
mv data-25m-add.txt ../datasets/.


cargo run --release -- --min 3 --max 5 --size 100000000 --seed 32 --save-file-path data-100m-all.txt
mv data-100m-all.txt ../datasets/.

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