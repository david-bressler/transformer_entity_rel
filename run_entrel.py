from entrel import EntityRelevance
entreller=EntityRelevance()


runtypea=1

if runtypea==0:
    entreller.explore_data() #look at the data
    entreller.create_datasets() #create datasets
if runtypea==1:
    results=entreller.train_one('company_test')