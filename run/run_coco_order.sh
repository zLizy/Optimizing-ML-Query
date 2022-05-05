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
data=coco_voc
constraint=cost
for type in dnf cnf;
do
    # f1 recall ap
    for metric in f1 recall ap;
    do
        #for coverage in 20_prob_0.85 30_prob_0.85 20_prob_0.7 30_prob_0.7;
        # 10 15 20 25 
        for coverage in 30;
        do
            # medium, low
            for level in high;
            do
                
                addition=${metric}_${type}_${constraint}_${coverage}_${level}
                chmod +x script_new_inference/${data}_${addition}/*.sh
                # files=(`ls script/coco${addition}/script_baseline*`)
                files=(`ls script_new_inference/${data}_${addition}/script_*`)
                for x in ${files[@]}
                do
                    echo ${x}
                    ./${x}
                done
            done
        done
    done
done