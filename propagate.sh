for slice in weimar; do \
for dom in sentiment patriotic economic conspiratorial religious racist ethic; do \
python propagate.py \
	--ppmi matrices/ppmi_${slice}.npz \
	--index ppmi_vocab/${slice}.json \
	--protocol_type RT \
	--semantic_domain $dom \
	--output_file ${slice}
done
done
