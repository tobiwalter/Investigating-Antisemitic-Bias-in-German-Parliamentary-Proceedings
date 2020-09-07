for slice in cdu_1 spd_1 cdu_2 spd_2 cdu_3; do \
python simlex_test.py \
	--vocab_file_pattern ../data/vocab/${slice}.json \
	--vector_file_pattern ../models/${slice}.vectors.npy \
	--output_file ${slice}.txt
done

	


