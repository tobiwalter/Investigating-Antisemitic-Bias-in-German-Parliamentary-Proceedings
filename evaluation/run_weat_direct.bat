FOR %%A IN (1,2,3,4,6) DO ^
FOR %%B IN (1) DO ^
python weat_adapted.py ^
			--test_number %%A ^
			--permutation_number 1000000 ^
			--lower False ^
			--is_vec_format True ^
			--embeddings ^
			../../obj/wp%%B.txt ^
			--output_file ^
			../output/weat%%A_wp%%1.txt
