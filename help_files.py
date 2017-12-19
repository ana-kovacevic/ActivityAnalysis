import pandas as pd


#Create pandas row

dict={'AIC':234234, 'BIC':[23, 44, 55]}

dict2={'aic':{'v1':12, 'v2':123}}

a={'Activity':'sadasd', 'BIC':12.312, 'AIC':23234}
b={'Activity':'aa', 'BIC':12, 'AIC':234}
li = []
li.append(a)
li.append(b)

c=pd.DataFrame(li)
c
c.to_csv('ana.csv')

type(c)

