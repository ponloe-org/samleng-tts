#!/bin/zsh
# Score a run's steps on seeds 0-2, pick its best step, then let that single model challenge the
# champion on fresh seeds 3-5 (both models, Latin and Khmer-script input, scale 1.2).
#   ops/ladder_round.sh RUN "LABEL" STEP...
export PATH=$HOME/.local/bin:/opt/homebrew/bin:$PATH
cd ~/code/samleng-tts
RUN=${1:?usage: ladder_round.sh RUN LABEL STEP...} LABEL=$2
shift 2
STEPS=($@)
while pgrep -f "watch_run.sh outputs/train/$RUN" >/dev/null; do sleep 30; done
SYN="--noise-scale-dp 0 --length-scale 1.2"
D=$(ls -td outputs/train/$RUN/*/(N) | while read d; do ls $d/checkpoint_*.pth >/dev/null 2>&1 && { print ${d%/}; break }; done)
[[ -n $D ]] || { print "no checkpoints for $RUN"; exit 1 }
for s in $STEPS; do
  [[ -e $D/checkpoint_$s.pth ]] || ln -s ../../keep/$RUN/checkpoint_$s.pth $D/checkpoint_$s.pth
  [[ -e $D/checkpoint_$s.pth ]] || { print "missing $RUN checkpoint $s"; exit 1 }
done
pick() {  # pick DIR SEEDS STEP...
  for t in "" --translit; do
    uv run python scripts/pick_checkpoint.py $1 --steps ${@:3} --seeds $2 $t --voice teacher_calm/0 --pace 0.7 "--synth=$SYN" > /dev/null || exit 1
  done
}
pick $D 3 $STEPS
B=$(uv run python scripts/ladder.py best $RUN --steps $STEPS) || exit 1
pick $D 6 $B
C=($(uv run python scripts/ladder.py champion)) || exit 1
pick $C[3] 6 $C[2]
CK=outputs/train/keep/$RUN/checkpoint_$B.pth
[[ -e $CK ]] || CK=$D/checkpoint_$B.pth
uv run python scripts/ladder.py challenge $RUN --step $B --run-dir $D --checkpoint $CK --label "$LABEL" || exit 1
# Free the disk (a run leaves ~10 GB): keep only the scored steps, plus the start point for later runs.
for f in $D/*.pth(N.); do [[ $f -ef $CK ]] || rm -f $f; done
for f in outputs/train/keep/$RUN/checkpoint_*.pth(N); do
  s=${${f:t:r}#checkpoint_}
  (( ${STEPS[(Ie)$s]} )) || rm -f $f
done
df -h . | tail -1
