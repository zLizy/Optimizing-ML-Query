# run script of preprocessing

cd evaluation
# Model inference and results recording
python3 inference.py
python3 inference.py -test
# Evaluate the performance of models and generate model repository performance tables
python3 getModelRepository.py
cd ..

./run/get_plans.sh

