# CS2 Demo Analyzer

Parser pipeline for CS2 `.dem` files that builds sequence-model ML samples and writes them to MongoDB.

## Install

```bash
python3 -m pip install -e '.[all]'
```

`demoparser2`, `pymongo`, and `requests` are optional extras so the package can still be inspected without native parser dependencies installed.

## Run

```bash
python3 -m cs2_demo_parser --config config.parse.json --input-mode local
```

Useful overrides:

```bash
python3 -m cs2_demo_parser \
  --config config.parse.json \
  --input-mode local \
  --max-workers 4 \
  --sample-every-ticks 16 \
  --context-window-size 128 \
  --context-window-count 8 \
  --log-level INFO
```

Quiet run with only the main per-demo progress bar:

```bash
python3 -m cs2_demo_parser \
  --config config.parse.json \
  --input-mode local \
  --max-workers 2 \
  --log-level WARNING \
  --quiet-progress
```

Offline/local Mongo run on the laptop:

```bash
python3 -m cs2_demo_parser \
  --config config.parse.local.json \
  --input-mode local \
  --max-workers 2 \
  --log-level WARNING \
  --quiet-progress
```

`config.parse.local.json` writes to `mongodb://localhost:27017/cs2_demo_ml` and keeps its own
state file at `state/processed_matches.local.json`. It still deletes local `.dem` files only after a
successful Mongo write.

Demos that parse but fail dataset quality checks, for example suspiciously low round count or zero
ML samples, are moved to `demos/rejected` and recorded under `failed` in the state file. Mongo/network
write failures still abort the run so a broken write is not marked as processed.

When the laptop is back on the home LAN, sync the local parsed matches to the home MongoDB:

```bash
python3 -m cs2_demo_parser.sync_mongo \
  --source-uri mongodb://localhost:27017 \
  --target-uri mongodb://192.168.1.106:27017 \
  --db cs2_demo_ml
```

To move data instead of leaving a laptop-local copy, use the default local-to-home direction:

```bash
python3 -m cs2_demo_parser.sync_mongo --move
```

For larger transfers over LAN, the sync uses batches of 200 samples by default. You can tune it:

```bash
python3 -m cs2_demo_parser.sync_mongo --move --batch-size 200
```

The sync is idempotent by `match_id`: complete matches that already exist on the target are skipped.
For each copied match it deletes any partial target samples for that match, copies all local samples,
verifies the sample count, then writes the target `ml_matches` document. With `--move`, source
documents are deleted from laptop-local Mongo only after the target copy is verified. Use `--dry-run`
to preview what would be copied.

## Sample representation

The default dataset preset is aimed at a first sequential VAE/CVAE-style world model:

- `sample_every_ticks=16`: one model timestep per roughly 125 ms on 128 tick demos.
- `history_steps=64`: about 8 seconds of context.
- `future_steps=32`: about 4 seconds of future target.
- `sample_stride_steps=4`: create a training sample every 4 sampled timesteps, roughly every 0.5 s.
- `stride_steps=1`: keep consecutive timesteps inside the history/future windows.

Each `ml_samples` document uses schema version 2:

- `history`: `[64, 10, player_feature_count]`
- `history_mask`: `[64, 10]`
- `future`: `[32, 10, target_feature_count]`, currently target positions by default.
- `future_mask`: `[32, 10]`
- `future_alive`: `[32, 10, 1]`
- `global_features`: round timing, team alive/HP/armor/economy summaries, bomb state and recent events.
- `labels`: round winner plus future event labels such as kill/plant/defuse/explosion.

Use `MongoSequenceSampleStore` from `cs2_demo_parser.model_samples` in training code to stream these documents from Mongo.

## Train World Model

After parsing demos into MongoDB, train the first self-supervised CVAE world model:

```bash
python3 -m cs2_demo_parser.train_world_model --config config.train.json
```

For a quick smoke run:

```bash
python3 -m cs2_demo_parser.train_world_model \
  --config config.train.json \
  --epochs 1 \
  --batch-size 8 \
  --limit-train-batches 10 \
  --limit-val-batches 2
```

To log training to Weights & Biases, first authenticate once:

```bash
wandb login
```

Then enable W&B for a run:

```bash
python3 -m cs2_demo_parser.train_world_model \
  --config config.train.json \
  --epochs 5 \
  --batch-size 4 \
  --mongo-batch-size 8 \
  --checkpoint-dir checkpoints/world_model_first \
  --wandb \
  --wandb-project cs2-demo-world-model \
  --wandb-run-name first-world-model
```

For a run that logs locally and can be synced later:

```bash
python3 -m cs2_demo_parser.train_world_model \
  --config config.train.json \
  --epochs 5 \
  --batch-size 4 \
  --mongo-batch-size 8 \
  --checkpoint-dir checkpoints/world_model_first \
  --wandb \
  --wandb-mode offline
```

For faster and more stable training, export Mongo samples to local PyTorch shards first:

```bash
python3 -m cs2_demo_parser.export_training_shards \
  --config config.train.local.json \
  --output-dir data/world_model_shards \
  --shard-size 4096 \
  --mongo-batch-size 32 \
  --overwrite
```

Then train from the shard directory instead of MongoDB:

```bash
python3 -m cs2_demo_parser.train_world_model \
  --config config.train.local.json \
  --shard-dir data/world_model_shards \
  --epochs 5 \
  --batch-size 16 \
  --num-workers 2 \
  --checkpoint-dir checkpoints/world_model_sharded \
  --wandb \
  --wandb-project cs2-demo-world-model \
  --wandb-run-name first-world-model-sharded
```

On Windows PowerShell, use backticks:

```powershell
python -m cs2_demo_parser.export_training_shards `
  --config config.train.local.json `
  --output-dir data/world_model_shards `
  --shard-size 4096 `
  --mongo-batch-size 32 `
  --overwrite

python -m cs2_demo_parser.train_world_model `
  --config config.train.local.json `
  --shard-dir data/world_model_shards `
  --epochs 5 `
  --batch-size 16 `
  --num-workers 2 `
  --checkpoint-dir checkpoints/world_model_sharded `
  --wandb `
  --wandb-project cs2-demo-world-model `
  --wandb-entity kgurgul-politechnika-wroc-awska `
  --wandb-run-name first-world-model-sharded
```

The model is implemented in `cs2_demo_parser.world_model.WorldModelCVAE`. It learns a latent `z`
conditioned on history/global state and predicts future positions, future alive masks, future event
labels and the round winner. Checkpoints are written to `checkpoints/world_model/latest.pt` and
`checkpoints/world_model/best.pt`.

To train against the laptop-local MongoDB, use:

```bash
python3 -m cs2_demo_parser.train_world_model --config config.train.local.json
```

## Mongo collections

- `ml_matches`: one document per match, including metadata, players, rounds and parse metadata.
- `ml_samples`: sequence samples with `history`, `future`, masks, global features, `labels` and `round_winner`.
- `ml_meta`: dataset configuration, feature lists, normalization stats and categorical vocabularies.

The writer creates indexes for match idempotency and sample lookup, removes old samples for a match before rewrite, then records the match as processed in `state/processed_matches.json`. Local `.dem` files are deleted only after Mongo writes and state update complete successfully.
