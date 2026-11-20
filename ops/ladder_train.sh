#!/bin/zsh
# Train one ladder run on the calm data, resuming from its newest checkpoint; the watcher scores every 2000 at scale 1.2.
#   ops/ladder_train.sh NAME TARGET_STEPS MANIFEST [train.py options, e.g. --init CKPT --lr-final 0.1]
# The run stops ~2,000 steps past TARGET_STEPS so the last scored step is safely there.
export PATH=$HOME/.local/bin:/opt/homebrew/bin:$PATH
cd ~/code/samleng-tts
NAME=${1:?usage: ladder_train.sh NAME TARGET_STEPS MANIFEST [options]} TARGET=$2 SRC=$3
shift 3
OUT=outputs/train/$NAME
mkdir -p $OUT outputs/logs
[[ -f $OUT/manifest.jsonl ]] || cp $SRC $OUT/manifest.jsonl || exit 1
# Bucketing gives ~3% fewer steps per pass than rows/16.
N=$(wc -l < $OUT/manifest.jsonl)
EPOCHS=$(( ((TARGET + 2000) * 16 + N - 101) / (N - 100) ))
print -r -- "$NAME: rows $N, epochs $EPOCHS, options: $*"
pgrep -f "watch_run.sh $OUT" >/dev/null || nohup /bin/zsh ops/watch_run.sh $OUT --voice teacher_calm/0 --pace 0.7 --both "--synth=--noise-scale-dp 0 --length-scale 1.2" >> outputs/logs/watch_$NAME.log 2>&1 < /dev/null &
cd runners/train_vits
RUN=$(ls -td ../../$OUT/*/(N) | while read d; do ls $d/checkpoint_*.pth >/dev/null 2>&1 && { print ${d%/}; break }; done)
if [[ -n $RUN ]]; then
  uv run python train.py ../../$OUT/manifest.jsonl ../../$OUT --continue-run $RUN --device mps --epochs $EPOCHS --save-step 1000 --workers 4 --eval-workers 0
else
  uv run python train.py ../../$OUT/manifest.jsonl ../../$OUT --device mps --batch-size 16 --epochs $EPOCHS --eval-clips 100 --save-step 1000 --workers 4 --eval-workers 0 "$@"
fi
