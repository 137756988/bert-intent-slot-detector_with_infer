#!/bin/bash
# 金融意图识别模型训练启动脚本

# # 检查Poetry是否安装
# if ! command -v poetry &> /dev/null; then
#     echo "Poetry未安装，正在安装..."
#     pip install poetry
# fi

# # 安装依赖
# echo "正在安装项目依赖..."
# poetry install


# 开始训练
echo "开始模型训练..."
# python train.py \
#     --cuda_devices 0 \
#     --tokenizer_path "bert-base-chinese" \
#     --model_path "bert-base-chinese" \
#     --train_data_path "data/FinQA/train_FinQA.json" \
#     --test_data_path "data/FinQA/test_FinQA.json" \
#     --intent_label_path "data/FinQA/intent_labels_FinQA.txt" \
#     --slot_label_path "data/FinQA/slot_labels_FinQA.txt" \
#     --save_dir "/output_model/FinQA_20250518_v1" \
#     --train_epochs 5 \
#     --learning_rate 5e-5 \
#     --batch_size 16 \

# 使用中文BERT-wwm-ext模型（推荐）
# python train.py \
#        --cuda_devices 0 \
#        --tokenizer_path "hfl/chinese-bert-wwm-ext" \
#        --model_path "hfl/chinese-bert-wwm-ext" \
#        --train_data_path "data/FinQA/train_FinQA.json" \
#        --test_data_path "data/FinQA/test_FinQA.json" \
#        --intent_label_path "data/FinQA/intent_labels_FinQA.txt" \
#        --slot_label_path "data/FinQA/slot_labels_FinQA.txt" \
#        --save_dir "output_model/FinQA_wwm_ext_20250518_v2" \
#        --batch_size 16 \
#        --train_epochs 5

# 使用哈工大RoBERTa-wwm-ext-large模型（推荐）
python train.py \
       --cuda_devices 0 \
       --tokenizer_path "hfl/chinese-roberta-wwm-ext-large" \
       --model_path "hfl/chinese-roberta-wwm-ext-large" \
       --train_data_path "data/FinQA/train_FinQA.json" \
       --test_data_path "data/FinQA/test_FinQA.json" \
       --intent_label_path "data/FinQA/intent_labels_FinQA.txt" \
       --slot_label_path "data/FinQA/slot_labels_FinQA.txt" \
       --save_dir "output_model/FinQA_roberta_wwm_ext_large_20250518_v1" \
       --batch_size 16 \
       --train_epochs 5



# 使用华为Noah中文预训练模型
# python train.py \
#        --cuda_devices 0 \
#        --tokenizer_path "IDEA-CCNL/Erlangshen-MegatronBert-1.3B" \
#        --model_path "IDEA-CCNL/Erlangshen-MegatronBert-1.3B" \
#        --train_data_path "data/FinQA/train_FinQA.json" \
#        --test_data_path "data/FinQA/test_FinQA.json" \
#        --intent_label_path "data/FinQA/intent_labels_FinQA.txt" \
#        --slot_label_path "data/FinQA/slot_labels_FinQA.txt" \
#        --save_dir "output_model/FinQA_erlangshen_20250518_v1" \
#        --batch_size 16 \
#        --train_epochs 5

# 使用BERT-base-multilingual-cased模型
# python train.py \
#        --cuda_devices 0 \
#        --tokenizer_path "bert-base-multilingual-cased" \
#        --model_path "bert-base-multilingual-cased" \
#        --train_data_path "data/FinQA/train_FinQA.json" \
#        --test_data_path "data/FinQA/test_FinQA.json" \
#        --intent_label_path "data/FinQA/intent_labels_FinQA.txt" \
#        --slot_label_path "data/FinQA/slot_labels_FinQA.txt" \
#        --save_dir "output_model/FinQA_multilingual_20250518_v1" \
#        --batch_size 16 \
#        --train_epochs 5

# 使用哈工大RoBERTa-wwm-ext模型
# python train.py \
#        --cuda_devices 0 \
#        --tokenizer_path "hfl/chinese-roberta-wwm-ext" \
#        --model_path "hfl/chinese-roberta-wwm-ext" \
#        --train_data_path "data/FinQA/train_FinQA.json" \
#        --test_data_path "data/FinQA/test_FinQA.json" \
#        --intent_label_path "data/FinQA/intent_labels_FinQA.txt" \
#        --slot_label_path "data/FinQA/slot_labels_FinQA.txt" \
#        --save_dir "output_model/FinQA_roberta_wwm_ext_20250518_v1" \
#        --batch_size 16 \
#        --train_epochs 5

# 使用阿里ALBERT-tiny模型（轻量级）
# python train.py \
#        --cuda_devices 0 \
#        --tokenizer_path "voidful/albert_chinese_tiny" \
#        --model_path "voidful/albert_chinese_tiny" \
#        --train_data_path "data/FinQA/train_FinQA.json" \
#        --test_data_path "data/FinQA/test_FinQA.json" \
#        --intent_label_path "data/FinQA/intent_labels_FinQA.txt" \
#        --slot_label_path "data/FinQA/slot_labels_FinQA.txt" \
#        --save_dir "output_model/FinQA_albert_tiny_20250518_v1" \
#        --batch_size 32 \
#        --train_epochs 8

# 使用金融领域预训练模型FinBERT
# python train.py \
#        --cuda_devices 0 \
#        --tokenizer_path "yiyanghkust/finbert-tone" \
#        --model_path "yiyanghkust/finbert-tone" \
#        --train_data_path "data/FinQA/train_FinQA.json" \
#        --test_data_path "data/FinQA/test_FinQA.json" \
#        --intent_label_path "data/FinQA/intent_labels_FinQA.txt" \
#        --slot_label_path "data/FinQA/slot_labels_FinQA.txt" \
#        --save_dir "output_model/FinQA_finbert_20250518_v1" \
#        --batch_size 16 \
#        --train_epochs 5

# 使用中文ELECTRA模型
# python train.py \
#        --cuda_devices 0 \
#        --tokenizer_path "hfl/chinese-electra-180g-large-discriminator" \
#        --model_path "hfl/chinese-electra-180g-large-discriminator" \
#        --train_data_path "data/FinQA/train_FinQA.json" \
#        --test_data_path "data/FinQA/test_FinQA.json" \
#        --intent_label_path "data/FinQA/intent_labels_FinQA.txt" \
#        --slot_label_path "data/FinQA/slot_labels_FinQA.txt" \
#        --save_dir "output_model/FinQA_electra_large_20250518_v1" \
#        --batch_size 8 \
#        --train_epochs 5

echo "训练完成，模型已保存到 output_model 目录下"