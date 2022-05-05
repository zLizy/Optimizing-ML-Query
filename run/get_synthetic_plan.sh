# SYNTHETIC
outdir=synthetic_cost
q_file=simulation/synthetic_query_uniform.csv
bound=100
for query_idx in {0..9}
do	
if [ $query_idx == 19 ] || [ $query_idx == 9 ] || [ $query_idx == 8 ]; then bound=10; fi
python3 query_optimizer.py -mdist uniform -qdist uniform -synthetic -record -qfile ${q_file} -outdir ${outdir} -constraint cost -bound ${bound} -approach baseline
python3 query_optimizer.py -mdist uniform -qdist uniform -synthetic -record -qfile ${q_file} -outdir ${outdir} -constraint cost -bound ${bound}
python3 query_optimizer.py -mdist uniform -qdist uniform -synthetic -record -qfile ${q_file} -outdir ${outdir} -constraint cost -bound ${bound} -order
done

outdir=synthetic_accu
bound=0.9
for query_idx in {0..9}
do	
	if [ $query_idx == 19 ] || [ $query_idx == 9 ] || [ $query_idx == 8 ]; then bound=10; fi
python3 query_optimizer.py -mdist uniform -qdist uniform -synthetic -record -qfile ${q_file} -outdir ${outdir} -constraint accuracy -bound ${bound} -approach baseline
python3 query_optimizer.py -mdist uniform -qdist uniform -synthetic -record -qfile ${q_file} -outdir ${outdir} -constraint accuracy -bound ${bound}
python3 query_optimizer.py -mdist uniform -qdist uniform -synthetic -record -qfile ${q_file} -query-idx ${query_idx} -outdir ${outdir} -constraint accuracy -bound ${bound} -order

done