for slice in cdu_1 spd_1 cdu_2 spd_2 cdu_3; do \
python ect_and_bat.py \
	--test_type BAT \
	--protocol_type BRD \
	--output_file /bat/${slice}.txt \
	--vocab_file_pattern ${slice}.json \
	--vector_file_pattern ${slice}.vectors.npy
done;


