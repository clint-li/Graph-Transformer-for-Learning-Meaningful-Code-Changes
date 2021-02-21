#!/usr/bin/env bash

function make_dir () {
    if [[ ! -d "$1" ]]; then
        mkdir $1
    fi
}

SRC_DIR=../..
DATA_DIR=${SRC_DIR}/data
MODEL_DIR=${SRC_DIR}/$4

make_dir $MODEL_DIR

DATASET=java
CODE_EXTENSION=buggy.txt
JAVADOC_EXTENSION=fixed.txt
DATA_PATH=$3


function train () {

echo "============TRAINING============"

RGPU=$1
MODEL_NAME=$2


PYTHONPATH=$SRC_DIR CUDA_VISIBLE_DEVICES=$RGPU python -W ignore ${SRC_DIR}/main/train.py \
--data_workers 0 \
--dataset_name $DATASET \
--data_dir ${DATA_DIR}/ \
--model_dir $MODEL_DIR \
--model_name $MODEL_NAME \
--train_src $DATA_PATH/train/${CODE_EXTENSION} \
--train_tgt $DATA_PATH/train/${JAVADOC_EXTENSION} \
--dev_src $DATA_PATH/dev/${CODE_EXTENSION} \
--dev_tgt $DATA_PATH/dev/${JAVADOC_EXTENSION} \
--uncase True \
--use_src_word True \
--use_src_char False \
--use_tgt_word True \
--use_tgt_char False \
--max_src_len 100 \
--max_tgt_len 50 \
--emsize 512 \
--fix_embeddings False \
--src_vocab_size 10000 \
--tgt_vocab_size 10000 \
--share_decoder_embeddings True \
--max_examples -1 \
--batch_size 16 \
--test_batch_size 16 \
--num_epochs 240 \
--model_type transformer \
--num_head 8 \
--d_k 64 \
--d_v 64 \
--d_ff 1024 \
--src_pos_emb False \
--tgt_pos_emb True \
--max_relative_pos 32 \
--use_neg_dist True \
--nlayers 6 \
--trans_drop 0.1 \
--dropout_emb 0.2 \
--dropout 0.1 \
--copy_attn True \
--early_stop 300 \
--warmup_steps 2000 \
--optimizer adam \
--learning_rate 0.0001 \
--lr_decay 0.99 \
--valid_metric ml_loss \
--checkpoint True

}

function test () {

echo "============TESTING============"

RGPU=$1
MODEL_NAME=$2

PYTHONPATH=$SRC_DIR CUDA_VISIBLE_DEVICES=$RGPU python -W ignore ${SRC_DIR}/main/train.py \
--only_test True \
--data_workers 0 \
--dataset_name $DATASET \
--data_dir ${DATA_DIR}/ \
--model_dir $MODEL_DIR \
--model_name $MODEL_NAME \
--dev_src $DATA_PATH/test/${CODE_EXTENSION} \
--dev_tgt $DATA_PATH/test/${JAVADOC_EXTENSION} \
--uncase True \
--max_src_len 100 \
--max_tgt_len 50 \
--max_examples -1 \
--test_batch_size 16

}

function beam_search () {

echo "============Beam Search TESTING============"

RGPU=$1
MODEL_NAME=$2

PYTHONPATH=$SRC_DIR CUDA_VISIBLE_DEVICES=$RGPU python -W ignore ${SRC_DIR}/main/test.py \
--data_workers 0 \
--dataset_name $DATASET \
--data_dir ${DATA_DIR}/ \
--model_dir $MODEL_DIR \
--model_name $MODEL_NAME \
--dev_src $DATA_PATH/test/${CODE_EXTENSION} \
--dev_tgt $DATA_PATH/test/${JAVADOC_EXTENSION} \
--uncase True \
--max_examples -1 \
--max_src_len 100 \
--max_tgt_len 50 \
--test_batch_size 16 \
--beam_size 5 \
--n_best 5 \
--block_ngram_repeat 3 \
--stepwise_penalty False \
--coverage_penalty none \
--length_penalty none \
--beta 0 \
--gamma 0 \
--replace_unk

}

train $1 $2
test $1 $2
beam_search $1 $2


echo "============CLASSIFICATION============="

RGPU=$1
MODEL_NAME=$2

export TEST_TARGETS=$MODEL_DIR/${MODEL_NAME}_fixed.txt
export TEST_SOURCES=$MODEL_DIR/${MODEL_NAME}_buggy.txt
export PREDICTIONS=$MODEL_DIR/${MODEL_NAME}
total=`wc -l ${TEST_TARGETS}| awk '{print $1}'`

echo "Test Set: $total"

echo "---Predictions---"
output=$(python prediction_classifier.py ${TEST_SOURCES} ${TEST_TARGETS} "${PREDICTIONS}_test_predictions.txt" 2>&1)

perf=`awk '{print $1}' <<< "$output"`
changed=`awk '{print $2}' <<< "$output"`
bad=`awk '{print $3}' <<< "$output"`
perf_perc="$(echo "$perf / $total" )"

echo "Perf: $perf"
echo "Pot : $changed"
echo "Bad : $bad"


echo "---Beam Predictions---"
output=$(python beam_prediction_classifier.py ${TEST_SOURCES} ${TEST_TARGETS} "${PREDICTIONS}_beam_predictions.txt" 2>&1)

perf=`awk '{print $1}' <<< "$output"`
changed=`awk '{print $2}' <<< "$output"`
bad=`awk '{print $3}' <<< "$output"`
perf_perc="$(echo "$perf / $total" )"

echo "Perf: $perf"
echo "Pot : $changed"
echo "Bad : $bad"