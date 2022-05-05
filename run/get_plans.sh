# python3 query_optimizer.py -constraint accuracy -check -outdir synthetic_32 -synthetic -approach baseline

# REAL
# repo=model_stats_ap_20.csv
# outdir=coco_ap_cnf_cost
# type=cnf

# for query_idx in 1 3;
# do

# 	# for i in 0 1 2 3 4;
# 	for bound in 0.7 0.75;
# 	do
# 		# bound=$(($base*$i/2+$base))
# 		echo $bound
# 		python3 query_optimizer.py -constraint accuracy -bound ${bound} -record -qtype ${type} -outdir ${outdir} -repo ${repo} -query-idx ${query_idx}
# 		python3 query_optimizer.py -constraint accuracy -bound ${bound} -record -qtype ${type} -outdir ${outdir} -repo ${repo} -query-idx ${query_idx} -order 
# 		python3 query_optimizer.py -constraint accuracy -bound ${bound} -record -qtype ${type} -outdir ${outdir} -repo ${repo} -query-idx ${query_idx} -approach baseline
# 	done
# done


# baseline as reference 
# +50%, +100%, +150%, +200%

# repo=model_stats_ap.csv
# outdir=coco_ap_cnf_accu
# type=cnf

# for query_idx in 1 3;
# do
# 	if [ $query_idx == 0 ] || [ $query_idx == 4 ]; then base=50; fi
# 	if [ $query_idx == 1 ]; then base=60; fi
# 	if [ $query_idx == 2 ]; then base=80; fi
# 	if [ $query_idx == 3 ]; then base=100; fi

# 	# for i in 0 1 2 3 4;
# 	for i in 0
# 	do
# 		bound=$(($base*$i/2+$base))
# 		echo $bound
# 		# python3 query_optimizer.py -constraint cost -bound ${bound} -record -qtype ${type} -outdir ${outdir} -repo ${repo} -query-idx ${query_idx}
# 		# python3 query_optimizer.py -constraint cost -bound ${bound} -record -qtype ${type} -outdir ${outdir} -repo ${repo} -query-idx ${query_idx} -order 
# 		# python3 query_optimizer.py -constraint cost -bound ${bound} -record -qtype ${type} -outdir ${outdir} -repo ${repo} -query-idx ${query_idx} -approach baseline
# 	done
# done


# repo=model_stats_ap.csv
# outdir=coco_ap_baseline_dnf_cost_1
# type=dnf

# for query_idx in 0 1 2 3 4;
# do
# 	if [ $query_idx == 1 ] || [ $query_idx == 3 ] || [ $query_idx == 4 ]; then base=80; fi
# 	if [ $query_idx == 0 ]; then base=30; fi
# 	if [ $query_idx == 2 ]; then base=50; fi

# 	for i in 0 1 2 3 4;
# 	do
# 		bound=$(($base*$i/2+$base))
# 		python3 query_optimizer.py -constraint cost -bound ${bound} -record -qtype ${type} -outdir ${outdir} -repo ${repo} -query-idx ${query_idx}
# 		python3 query_optimizer.py -constraint cost -bound ${bound} -record -qtype ${type} -outdir ${outdir} -repo ${repo} -query-idx ${query_idx} -order 
# 		python3 query_optimizer.py -constraint cost -bound ${bound} -record -qtype ${type} -outdir ${outdir} -repo ${repo} -query-idx ${query_idx} -approach baseline
# 	done
# done

# q_file=simulation/${data}_query_${type}_gt.csv
# python3 query_optimizer.py -check -data coco_voc -constraint cost -record -qtype ${type} -qfile ${q_file} \
# 		-outdir coco_ap_cnf -repo model_stats_ap.csv -approach baseline

# repo=model_stats_f1_new_model_20.csv
# outdir=coco_f1_cnf_cost_20
# type=cnf

# repo=model_stats_recall_new_model_30_prob_0.7.csv
# outdir=coco_recall_cnf_cost_30_prob_0.7

# cnf
# data=coco

# metric=precision
# coverage=30
# level=high

# repo=${data_type}_model_stats_${metric}_new_model_${coverage}_${level}.csv
# for query_idx in 1 3 4;
# do
# 	# for bound in 60 70 80 90 ;
# 	for bound in 40 60 80 100 120;
# 	do
# 		python3 query_optimizer.py -constraint cost -bound ${bound} -record -qtype ${type} -outdir ${outdir} -repo ${repo} -query-idx ${query_idx}
# 		python3 query_optimizer.py -constraint cost -bound ${bound} -record -qtype ${type} -outdir ${outdir} -repo ${repo} -query-idx ${query_idx} -order 
# 		python3 query_optimizer.py -constraint cost -bound ${bound} -record -qtype ${type} -outdir ${outdir} -repo ${repo} -query-idx ${query_idx} -approach baseline

# 	done
# done

# for query_idx in 0 2;
# do
# 	for bound in 30 45 60 75 90;
# 	do
# 		python3 query_optimizer.py -constraint cost -bound ${bound} -record -qtype ${type} -outdir ${outdir} -repo ${repo} -query-idx ${query_idx}
# 		python3 query_optimizer.py -constraint cost -bound ${bound} -record -qtype ${type} -outdir ${outdir} -repo ${repo} -query-idx ${query_idx} -order 
# 		python3 query_optimizer.py -constraint cost -bound ${bound} -record -qtype ${type} -outdir ${outdir} -repo ${repo} -query-idx ${query_idx} -approach baseline
# 	done
# done

###### original run script
data=coco_voc
data=voc
data_type=coco
data_type=voc
# constraint=cost
constraint=accuracy
level=high

# dnf
for type in cnf;
do
	# f1 recall precision accuracy
	for metric in f1 recall precision accuracy;
	do
		# 10 20
		for coverage in 30;
		do
			# pareto
			for distr in pareto;
			do
				echo ${distr}
				# repo = coco_model_stats_recall_new_model_30_high.csv
				# ${data_type}_model_stats_${metric}_pareto_${coverage}_${level}.csv
				repo=${data_type}_model_stats_${metric}_${distr}_${coverage}_${level}.csv
				# outdir = coco_voc_recall_cnf_accuracy
				if [ "$distr" = "pareto" ]; then
					outdir=${data}_${metric}_${type}_${constraint}_pareto
					scriptdir=script_pareto
					echo ${outdir}
				else
					outdir=${data}_${metric}_${type}_${constraint}
					scriptdir=script
				fi
				# qfile = simulation/coco_voc_query_cnf_gt.csv
				q_file=simulation/${data}_query_${type}_gt.csv
				
				# 1 2 3 4
				for query_idx in 0 1;
				do
					# for bound in 40 60 80 100 120;
					for bound in 0.7 0.9 0.95
					do
						python3 query_optimizer.py -data ${data_type} -constraint ${constraint} -bound ${bound} -record -qtype ${type} -qfile ${q_file} -outdir ${outdir} -repo ${repo} -query-idx ${query_idx} -scriptdir ${scriptdir}
						# -model-sel -record
						# order
						python3 query_optimizer.py -data ${data_type} -constraint ${constraint} -bound ${bound} -record -qtype ${type} -qfile ${q_file} -outdir ${outdir} -repo ${repo} -query-idx ${query_idx} -scriptdir ${scriptdir} -order 
						# greedy
						python3 query_optimizer.py -data ${data_type} -constraint ${constraint} -bound ${bound} -record -qtype ${type} -qfile ${q_file} -outdir ${outdir} -repo ${repo} -query-idx ${query_idx} -scriptdir ${scriptdir} -approach baseline

					done
				done
				for query_idx in 2 3;
				do
					# for bound in 60 90 120 150 180;
					for bound in 0.7 0.9 0.95
					do
						python3 query_optimizer.py -data ${data_type} -constraint ${constraint} -bound ${bound} -record -qtype ${type} -qfile ${q_file} -outdir ${outdir} -repo ${repo} -query-idx ${query_idx} -scriptdir ${scriptdir}
						# -model-sel
						# order
						python3 query_optimizer.py -data ${data_type} -constraint ${constraint} -bound ${bound} -record -qtype ${type} -qfile ${q_file} -outdir ${outdir} -repo ${repo} -query-idx ${query_idx} -scriptdir ${scriptdir} -order 
						python3 query_optimizer.py -data ${data_type} -constraint ${constraint} -bound ${bound} -record -qtype ${type} -qfile ${q_file} -outdir ${outdir} -repo ${repo} -query-idx ${query_idx} -scriptdir ${scriptdir} -approach baseline

					done
				done
				for query_idx in 4;
				do
					# for bound in 80 120 160 200 240;
					for bound in 0.7 0.9 0.95
					do
						python3 query_optimizer.py -data ${data_type} -constraint ${constraint} -bound ${bound} -record -qtype ${type} -qfile ${q_file} -outdir ${outdir} -repo ${repo} -query-idx ${query_idx} -scriptdir ${scriptdir}
						# order
						python3 query_optimizer.py -data ${data_type} -constraint ${constraint} -bound ${bound} -record -qtype ${type} -qfile ${q_file} -outdir ${outdir} -repo ${repo} -query-idx ${query_idx} -scriptdir ${scriptdir} -order 
						python3 query_optimizer.py -data ${data_type} -constraint ${constraint} -bound ${bound} -record -qtype ${type} -qfile ${q_file} -outdir ${outdir} -repo ${repo} -query-idx ${query_idx} -scriptdir ${scriptdir} -approach baseline

					done
				done
			done
		done
	done
done

###### external experiment run script
# metric=recall
# data=coco_2014
# data_type=coco
# constraint=cost
# qtype=dnf
# outdir=${data}_${metric}_${qtype}
# repo=${data}_model_stats_${metric}.csv
# for query in "orange|banana"
# # "banana&orange"
# do
# 	# for bound in 0.7 0.75 0.8 0.85 0.9 0.95 1.0;
# 	for bound in 35 45 55 65 75 85 95 
# 	do
# 		# coco_2014_model_stats_f1.csv
# 		python3 external_test.py -data ${data_type} -constraint ${constraint} -bound ${bound} \
# 		-qtype ${qtype} -outdir ${outdir} -record -repo ${repo} -query ${query}  -order -pp -model-sel
# 	done
# done

