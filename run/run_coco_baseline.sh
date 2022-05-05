# run-parts --regex '^s.*sh$' 
# run-parts --list --regex '^s.*sh$'
# find script/coco -type f -executable -name '*.sh' -exec {} \;


# addition='_f1_cnf_cost_20'
# config='model_config_new_model_20.csv'
# chmod +x script_new_model/coco${addition}/*.sh
# # files=(`ls script/coco${addition}/script_baseline*`)
# files=(`ls script_new_model/coco${addition}/script_baseline_1*`)
# for x in ${files[@]}
# do
# echo ${x}
# ./${x}
# done

# data=coco_voc
data=voc
constraint=accuracy
for type in cnf;
do
    # f1 recall accuracy precision
    for metric in accuracy precision;
    do
        # 10 20
        for coverage in 30;
        do
            # medium, low
            for level in high;
            do
                addition=${metric}_${type}_${constraint}_pareto
                # addition=${metric}_${type}_${constraint}
                chmod +x script_pareto/${data}_${addition}/*
                # files=(`ls script/${data}_${addition}/script_order*.sh`)
                files=(`ls script_pareto/${data}_${addition}/script_*.sh`)
                for x in ${files[@]}
                do
                    echo ${x}
                    ./${x}
                done
            done
        done
    done
done