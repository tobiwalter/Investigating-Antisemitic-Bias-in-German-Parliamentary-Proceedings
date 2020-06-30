for slice in cdu_1 spd_1 cdu_2 spd_2 cdu_3; do \
for i in $(seq 1 7);
	do python weat.py \
			--test_number $i \
			--protocol_type BT \
			--permutation_number 1000000 \
			--lower False \
			--is_vec_format False \
			--embedding_vectors \
			../models/${slice}.vectors.npy \
			--embedding_vocab \
			../data/vocab/${slice}.json \
			--output_file \
			output/weat${i}_${slice}.txt;
done;
done;
