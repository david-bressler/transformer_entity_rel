from entrel import EntityRelevance
entreller=EntityRelevance()


runtypea=1

if runtypea==0:
    entreller.explore_data() #look at the data
if runtypea==0.5: #explore and create datasets
    entreller.explore_data() #look at the data
    entreller.create_datasets() #create datasets
if runtypea==0.6: #explore and create datasets, mixed
    entreller.explore_data() #look at the data
    entreller.create_datasets_mixed() #create datasets
if runtypea==1: #run training for a single dataset
    results=entreller.train_one('company_person_fulltrainwtest_mixed')
    results=entreller.train_one('company_mixed')
    results=entreller.train_one('person_mixed')
if runtypea==2: #run evaluation on a single dataset
    results=entreller.eval_one('company_person_fulltrainwtest_mixed', 'company_person_test_mixed')
if runtypea==3: #run eval on mult datasets
    results=entreller.eval_mult(['company_person_fulltrainwtest_mixed','person_mixed','company_mixed'],['company_person_test_mixed','person_test_mixed','company_test_mixed'])
if runtypea==4: #hypersearch
    results=entreller.run_hypersearch('company_person_fulltrainwtest_mixed')  # 'company_person_fulltrainwtest')
if runtypea==5:
    entreller.analyze_hyper('outputta.pickle')
if runtypea==6:
    entreller.test_hyper_selection()