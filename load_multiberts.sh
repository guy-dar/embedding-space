mkdir multiberts
mkdir multiberts/models

wget -O multiberts/vocab.txt https://storage.googleapis.com/multiberts/public/vocab.txt
wget -O multiberts/bert_config.json https://storage.googleapis.com/multiberts/public/bert_config.json
for ckpt in {0..2} ; do
    wget "https://storage.googleapis.com/multiberts/public/models/seed_${ckpt}.zip" -O "multiberts/seed_${ckpt}.zip"
    unzip -o "multiberts/seed_${ckpt}.zip" -d "multiberts/models"
    rm "multiberts/seed_${ckpt}.zip"
    export BERT_BASE_DIR="multiberts/models/seed_${ckpt}"
    transformers-cli convert --model_type bert \
      --tf_checkpoint $BERT_BASE_DIR/bert.ckpt \
      --config multiberts/bert_config.json \
      --pytorch_dump_output $BERT_BASE_DIR/pytorch_model.bin
      
      cp multiberts/vocab.txt $BERT_BASE_DIR/vocab.txt
      cp multiberts/bert_config.json $BERT_BASE_DIR/config.json
done