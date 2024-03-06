export HCCL_CONNECT_TIMEOUT=600
export RANK_TABLE_FILE=/home/mindspore/zhangjunyuan/hccl_8p_01234567_127.0.0.1.json
export RANK_SIZE=8
export DEVICE_ID=0
export RANK_ID=0
export DEPLOY_MODE=0
export GE_USE_STATIC_MEMORY=1
rm -rf LOG0
mkdir ./LOG0
cp *.py ./LOG0
mkdir -p ./LOG0/ms_log
env > ./LOG0/env.log
export GLOG_log_dir=/home/mindspore/hhy/bert/models/official/nlp/Bert/LOG0/ms_log
export GLOG_logtostderr=0
cd /home/mindspore/hhy/bert/models/official/nlp/Bert/LOG0
taskset -c 0-23 nohup python /home/mindspore/hhy/bert/models/official/nlp/Bert/run_pretrain.py --distribute=true --epoch_size=40 --enable_save_ckpt=true --do_shuffle=true --enable_data_sink=true --data_sink_steps=100 --accumulation_steps=1 --allreduce_post_accumulation=true --save_checkpoint_path=./ --save_checkpoint_num=1 --config_path=../../pretrain_config.yaml --data_dir=/home/mindspore/hhy/dataset/chinese/ch_train --device_id=0 --device_num=8 >./pretraining_log.txt 2>&1 &
cd -
echo "run with rank_id=0 device_id=0 logic_id=0"

export DEVICE_ID=1
export RANK_ID=1
export DEPLOY_MODE=0
export GE_USE_STATIC_MEMORY=1
rm -rf LOG1
mkdir ./LOG1
cp *.py ./LOG1
mkdir -p ./LOG1/ms_log
env > ./LOG1/env.log
export GLOG_log_dir=/home/mindspore/hhy/bert/models/official/nlp/Bert/LOG1/ms_log
export GLOG_logtostderr=0
cd /home/mindspore/hhy/bert/models/official/nlp/Bert/LOG1
taskset -c 24-47 nohup python /home/mindspore/hhy/bert/models/official/nlp/Bert/run_pretrain.py --distribute=true --epoch_size=40 --enable_save_ckpt=true --do_shuffle=true --enable_data_sink=true --data_sink_steps=100 --accumulation_steps=1 --allreduce_post_accumulation=true --save_checkpoint_path=./ --save_checkpoint_num=1 --config_path=../../pretrain_config.yaml --data_dir=/home/mindspore/hhy/dataset/chinese/ch_train --device_id=1 --device_num=8 >./pretraining_log.txt 2>&1 &
cd -
echo "run with rank_id=1 device_id=1 logic_id=1"

export DEVICE_ID=2
export RANK_ID=2
export DEPLOY_MODE=0
export GE_USE_STATIC_MEMORY=1
rm -rf LOG2
mkdir ./LOG2
cp *.py ./LOG2
mkdir -p ./LOG2/ms_log
env > ./LOG2/env.log
export GLOG_log_dir=/home/mindspore/hhy/bert/models/official/nlp/Bert/LOG2/ms_log
export GLOG_logtostderr=0
cd /home/mindspore/hhy/bert/models/official/nlp/Bert/LOG2
taskset -c 48-71 nohup python /home/mindspore/hhy/bert/models/official/nlp/Bert/run_pretrain.py --distribute=true --epoch_size=40 --enable_save_ckpt=true --do_shuffle=true --enable_data_sink=true --data_sink_steps=100 --accumulation_steps=1 --allreduce_post_accumulation=true --save_checkpoint_path=./ --save_checkpoint_num=1 --config_path=../../pretrain_config.yaml --data_dir=/home/mindspore/hhy/dataset/chinese/ch_train --device_id=2 --device_num=8 >./pretraining_log.txt 2>&1 &
cd -
echo "run with rank_id=2 device_id=2 logic_id=2"

export DEVICE_ID=3
export RANK_ID=3
export DEPLOY_MODE=0
export GE_USE_STATIC_MEMORY=1
rm -rf LOG3
mkdir ./LOG3
cp *.py ./LOG3
mkdir -p ./LOG3/ms_log
env > ./LOG3/env.log
export GLOG_log_dir=/home/mindspore/hhy/bert/models/official/nlp/Bert/LOG3/ms_log
export GLOG_logtostderr=0
cd /home/mindspore/hhy/bert/models/official/nlp/Bert/LOG3
taskset -c 72-95 nohup python /home/mindspore/hhy/bert/models/official/nlp/Bert/run_pretrain.py --distribute=true --epoch_size=40 --enable_save_ckpt=true --do_shuffle=true --enable_data_sink=true --data_sink_steps=100 --accumulation_steps=1 --allreduce_post_accumulation=true --save_checkpoint_path=./ --save_checkpoint_num=1 --config_path=../../pretrain_config.yaml --data_dir=/home/mindspore/hhy/dataset/chinese/ch_train --device_id=3 --device_num=8 >./pretraining_log.txt 2>&1 &
cd -
echo "run with rank_id=3 device_id=3 logic_id=3"

export DEVICE_ID=4
export RANK_ID=4
export DEPLOY_MODE=0
export GE_USE_STATIC_MEMORY=1
rm -rf LOG4
mkdir ./LOG4
cp *.py ./LOG4
mkdir -p ./LOG4/ms_log
env > ./LOG4/env.log
export GLOG_log_dir=/home/mindspore/hhy/bert/models/official/nlp/Bert/LOG4/ms_log
export GLOG_logtostderr=0
cd /home/mindspore/hhy/bert/models/official/nlp/Bert/LOG4
taskset -c 96-119 nohup python /home/mindspore/hhy/bert/models/official/nlp/Bert/run_pretrain.py --distribute=true --epoch_size=40 --enable_save_ckpt=true --do_shuffle=true --enable_data_sink=true --data_sink_steps=100 --accumulation_steps=1 --allreduce_post_accumulation=true --save_checkpoint_path=./ --save_checkpoint_num=1 --config_path=../../pretrain_config.yaml --data_dir=/home/mindspore/hhy/dataset/chinese/ch_train --device_id=4 --device_num=8 >./pretraining_log.txt 2>&1 &
cd -
echo "run with rank_id=4 device_id=4 logic_id=4"

export DEVICE_ID=5
export RANK_ID=5
export DEPLOY_MODE=0
export GE_USE_STATIC_MEMORY=1
rm -rf LOG5
mkdir ./LOG5
cp *.py ./LOG5
mkdir -p ./LOG5/ms_log
env > ./LOG5/env.log
export GLOG_log_dir=/home/mindspore/hhy/bert/models/official/nlp/Bert/LOG5/ms_log
export GLOG_logtostderr=0
cd /home/mindspore/hhy/bert/models/official/nlp/Bert/LOG5
taskset -c 120-143 nohup python /home/mindspore/hhy/bert/models/official/nlp/Bert/run_pretrain.py --distribute=true --epoch_size=40 --enable_save_ckpt=true --do_shuffle=true --enable_data_sink=true --data_sink_steps=100 --accumulation_steps=1 --allreduce_post_accumulation=true --save_checkpoint_path=./ --save_checkpoint_num=1 --config_path=../../pretrain_config.yaml --data_dir=/home/mindspore/hhy/dataset/chinese/ch_train --device_id=5 --device_num=8 >./pretraining_log.txt 2>&1 &
cd -
echo "run with rank_id=5 device_id=5 logic_id=5"

export DEVICE_ID=6
export RANK_ID=6
export DEPLOY_MODE=0
export GE_USE_STATIC_MEMORY=1
rm -rf LOG6
mkdir ./LOG6
cp *.py ./LOG6
mkdir -p ./LOG6/ms_log
env > ./LOG6/env.log
export GLOG_log_dir=/home/mindspore/hhy/bert/models/official/nlp/Bert/LOG6/ms_log
export GLOG_logtostderr=0
cd /home/mindspore/hhy/bert/models/official/nlp/Bert/LOG6
taskset -c 144-167 nohup python /home/mindspore/hhy/bert/models/official/nlp/Bert/run_pretrain.py --distribute=true --epoch_size=40 --enable_save_ckpt=true --do_shuffle=true --enable_data_sink=true --data_sink_steps=100 --accumulation_steps=1 --allreduce_post_accumulation=true --save_checkpoint_path=./ --save_checkpoint_num=1 --config_path=../../pretrain_config.yaml --data_dir=/home/mindspore/hhy/dataset/chinese/ch_train --device_id=6 --device_num=8 >./pretraining_log.txt 2>&1 &
cd -
echo "run with rank_id=6 device_id=6 logic_id=6"

export DEVICE_ID=7
export RANK_ID=7
export DEPLOY_MODE=0
export GE_USE_STATIC_MEMORY=1
rm -rf LOG7
mkdir ./LOG7
cp *.py ./LOG7
mkdir -p ./LOG7/ms_log
env > ./LOG7/env.log
export GLOG_log_dir=/home/mindspore/hhy/bert/models/official/nlp/Bert/LOG7/ms_log
export GLOG_logtostderr=0
cd /home/mindspore/hhy/bert/models/official/nlp/Bert/LOG7
taskset -c 168-191 nohup python /home/mindspore/hhy/bert/models/official/nlp/Bert/run_pretrain.py --distribute=true --epoch_size=40 --enable_save_ckpt=true --do_shuffle=true --enable_data_sink=true --data_sink_steps=100 --accumulation_steps=1 --allreduce_post_accumulation=true --save_checkpoint_path=./ --save_checkpoint_num=1 --config_path=../../pretrain_config.yaml --data_dir=/home/mindspore/hhy/dataset/chinese/ch_train --device_id=7 --device_num=8 >./pretraining_log.txt 2>&1 &
cd -
echo "run with rank_id=7 device_id=7 logic_id=7"

