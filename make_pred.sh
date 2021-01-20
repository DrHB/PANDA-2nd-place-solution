cd train_rguo/train_code
python3 predict_rguo_part1.py
python3 predict_rguo_part2.py
cd ../../

cd train_cateek
python3 predict.py
cp data/submissions/submission.csv ../prediction/submission_cateek.csv
cd ../

cd train_drhb
python3 predict.py
cp submissions/submission.csv ../prediction/submission_drhb.csv
cd ../

cd train_xie29
python3 predict.py
cp submission/xie29_ensemble_submission.csv ../prediction/submission_xie29.csv
cd ../

python3 merge_prediction.py
