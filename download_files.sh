mkdir data
aws s3 cp s3://ai-ml-ebu-datasets/models/entity_relevance/roberta_entrel_20190107.tgz ./data/
tar -xvzf ./data/roberta_entrel_20190107.tgz -C ./data/
rm ./data/roberta_entrel_20190107.tgz
aws s3 cp s3://ai-ml-ebu-datasets/datasets/kyc-compliance/entity-relevance/people-relevance/preprocessed/contexts_samples_PERSON_test.json ./data/
aws s3 cp s3://ai-ml-ebu-datasets/datasets/kyc-compliance/entity-relevance/people-relevance/preprocessed/contexts_samples_person_test.json ./data/
aws s3 cp s3://ai-ml-ebu-datasets/datasets/kyc-compliance/entity-relevance/people-relevance/preprocessed/contexts_samples_PERSON_train.json ./data/
aws s3 cp s3://ai-ml-ebu-datasets/datasets/kyc-compliance/entity-relevance/people-relevance/preprocessed/contexts_samples_person_train.json ./data/
aws s3 cp s3://ai-ml-ebu-datasets/datasets/kyc-compliance/entity-relevance/company_relevance/contexts_samples_MT_train_new.json ./data/
aws s3 cp s3://ai-ml-ebu-datasets/datasets/kyc-compliance/entity-relevance/company_relevance/contexts_samples_MT_test_new.json ./data/
python -c "import nltk;nltk.download(\"punkt\");nltk.download(\"stopwords\")"
mkdir data/tmp
