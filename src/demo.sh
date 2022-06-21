# Quick run for KAO demo

# 1. Extract embeddings for the individuals specified in "data/individuals"
echo "Extracting individuals embeddings..."
python src/extract_model_embeddings.py demo tv_news --detector yolo --feats resnetv1
echo "...Done!"

# 2. Detect faces from the videos specified in "data/tv_news"
echo "Detecting faces from videos..."
python src/face_detection_kao.py demo tv_news --detector yolo --feats resnetv1 --extract_embs
echo "...Done!"

# 3. Assign IDs to the detected faces with respect to the specified individuals
echo "Classifying faces..."
python src/face_classifier_kao.py demo tv_news --detector yolo --feats resnetv1 --mod_feat fcg_average_vote
echo "...Done!"

# 4. Generate a video with the results (1fps)
echo "Generating video..."
python src/results_to_video.py demo tv_news --detector yolo --feat resnetv1 --mod_feat fcg_average_vote
echo "...Done!"
