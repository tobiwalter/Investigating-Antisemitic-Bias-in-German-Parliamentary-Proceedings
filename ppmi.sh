for slice in kaiserreich_1 kaiserreich_2 weimar; do \
python create_ppmi_mat.py \
	--protocols reichstag/${slice}_processed \
	--protocol_type RT \
	--min_count 10 \
	--window_size 2 \
	--output_file ${slice}
done
done
done
