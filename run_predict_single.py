from predict_single import *

my_predictor=Predictor()

#positive example:
stringa1='Accused Tanveer and Bilal Mir were brought from Kot Bhalwal Jail, Jammu, where they have been detained under Public Safety Act, \
     and produced before the NIA Special Court today against a production warrant,. the NIA said. The investigation has disclosed that earlier arrested \
          accused <BEGIN-SPAN> Kyocera <END-SPAN> was in regular contact with these accused over WhatsApp. the agency mentioned. The NIA court has granted \
              seven days custody of the two accused to the investigative agency (up to May 2)'

#negative example:
stringa2="scheme, according to Bloomberg. Buffett's Berkshire Hathaway conglomerate invested $340 million into Jeff and Paulette Carpoff's company, DC Solar, \
    according to a company filing. The scheme allegedly involved the sale of mobile solar generators to at least a dozen investors including insurance giant Progressive , \
    paintmaker <BEGIN-SPAN> Kyocera <END-SPAN> , and several regional banks , according to. The investors typically paid $45,000 of the $150,000 price tag for each of the \
        units up front, then claimed a $45,000 tax credit on their investment as well as tax deductions for the devices' depreciation, according to court filings. After the \
            FBI accused DC Solar of defrauding investors, Buffett and his team determined it was"

preds=my_predictor.predict(stringa1)
print(preds)

preds=my_predictor.predict(stringa2)
print(preds)
