import pandas as pd
import Quandl

### Features - Labels
### Quandl is a Stock price API 


df = Quandl.get('WIKI/GOOGL')
print(df.head)

df = df[['Adj. Open', 'Adj. High','Adj. Low','Adj. Close','Adj. Volume']]

