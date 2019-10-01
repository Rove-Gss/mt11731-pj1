set vocab=data/vocab.bin
set train_src=data/train.de-en.de.wmixerprep
set train_tgt=data/train.de-en.en.wmixerprep
set dev_src=data/valid.de-en.de
set dev_tgt=data/valid.de-en.en
set test_src=data/test.de-en.de
set test_tgt=data/test.de-en.en
set work_dir=work_dir

mkdir %work_dir%
echo save results to %work_dir%

python nmt.py train --cuda --vocab %vocab% --train-src %train_src% --train-tgt %train_tgt% --dev-src %dev_src% --dev-tgt %dev_tgt% --save-to %work_dir%/model.bin --valid-niter 2400 --batch-size 64 --hidden-size 256 --embed-size 256 --uniform-init 0.1 --dropout 0.2 --clip-grad 5.0  --lr-decay 0.5 --max-epoch 1
::2>%work_dir%/err.log

::python nmt.py decode --cuda --beam-size 5 --max-decoding-time-step 100 %work_dir%/model.bin %test_src% %work_dir%/decode.txt

::perl multi-bleu.perl %test_tgt% < %work_dir%/decode.txt