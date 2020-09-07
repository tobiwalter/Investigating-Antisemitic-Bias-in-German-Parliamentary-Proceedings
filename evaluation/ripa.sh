for slice in kaiserreich_1 kaiserreich_2 weimar; do \
for dom in sentiment patriotism economic conspiratorial religious racist ethic;
do python projections.py \
			--embedding_vocab \
			../data/vocab/${slice}.json \
			--protocol_type RT \
			--slice $slice \
			--sem_domain ${dom} \
			--embedding_vectors \
			../models/${slice}.vectors.npy \
			--plot_projections True \
			--t_test True \
			--plot_pca False \
			--output_file ${dom}_${slice}
done
done
