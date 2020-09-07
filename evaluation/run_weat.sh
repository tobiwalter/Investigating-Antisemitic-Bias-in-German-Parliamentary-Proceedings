for slice in kaiserreich_1 kaiserreich_2 weimar; do \
for i in 1; do \
for att in sentiment patriotism economic conspiratorial religious racist ethic; \
	do python weat.py \
			--test_number $i \
			--protocol_type RT \
			--permutation_number 12870 \
			--lower False \
			--sem_domain $att \
			--is_vec_format False \
			--embedding_vectors \
			../models/${slice}.vectors.npy \
			--embedding_vocab \
			../data/vocab/${slice}.json \
			--output_file \
			output/weat${i}_${att}_${slice}.txt;
done;
done;
done;
