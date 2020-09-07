for slice in cdu_1 spd_1 cdu_2 spd_2 cdu_3; do \
python kmeans_test.py \
	--vocab_file ../data/vocab/${slice}.json \
	--protocol_type BRD \
	--vector_file ../models/${slice}.vectors.npy
done

