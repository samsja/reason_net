# ReasonNet

how to run

```bash
python reason_net/run.py --config-path configs --config-name default.yaml module/model=70M
```


With small model for test

```bash
python reason_net/run.py --config-path configs --config-name default.yaml module/model=2M
```

generate 

```bash
python reason_net/generate.py lightning_logs/dummy-jv7tadve/last.ckpt 14M "1234+1234=" --interactive
```