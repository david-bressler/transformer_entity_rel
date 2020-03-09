from entrel import EntityRelevance
entreller=EntityRelevance()


runtypea=4

if runtypea==0.5:
    entreller.explore_data() #look at the data
if runtypea==0: #explore and create datasets
    entreller.explore_data() #look at the data
    entreller.create_datasets() #create datasets
if runtypea==1: #run training for a single dataset
    results=entreller.train_one('company_person_fulltrainwtest')
if runtypea==2: #run evaluation on a single dataset
    results=entreller.eval_one('company_person_fulltrainwtest', 'company_person_test')
if runtypea==3: #run eval on mult datasets
    results=entreller.eval_mult(['company_person_2ep','person_2ep','company_2ep'],['company_person_test','person_test','company_test'])
if runtypea==4: #hypersearch
    results=entreller.run_hypersearch('company_person_fulltrainwtest')  # 'company_person_fulltrainwtest')
if runtypea==5:
    entreller.analyze_hyper('outputta.pickle')
if runtypea==6:
    entreller.test_hyper_selection()