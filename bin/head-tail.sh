export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=$4

python3 test_token_bilstm.py  --input $1 --output $2  
#python3 test_token_many_bigram.py --input $2 --output $3
python3 test_token_many_tkbigram_one.py  --input $2 --output $3
