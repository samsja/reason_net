# ReasonNet


generate data 

```bash
cd data_gen
cargo run --release -- --min 3 --max 5 --size 20000000 --seed 32 --save-path data-20m-add --chunk-size=500000 --operators=+

mv data-20m-add ../datasets/.


cargo run --release -- --min 3 --max 5 --size 20000000 --seed 32 --save-path data-20m-all --chunk-size=500000 
mv data-20m-all ../datasets/.

```


run the training


normal 

```bash
python reason_net/run.py --config-path configs --config-name all-14m.yaml
```


reason middle

```bash
python reason_net/run.py --config-path configs --config-name all-14m.yaml reason_mode=true 
```


reason left 
```bash
python reason_net/run.py --config-path configs --config-name all-14m.yaml reason_mode=true +data.reason.reason_token_pos="left" 
```


With small model for test

```bash
python reason_net/run.py --config-path configs --config-name all-14m.yaml module/model=910K
```