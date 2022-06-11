############ PRICE OPTIMIZATION FOR RETAIL TO MAXIMIZE PROFIT ###############

# Importing data From Postgresql 
import psycopg2
conn=psycopg2.connect(dbname='Price_optimization',user='postgres',password='Bu6LYvqQw@123',host='127.0.0.1',port='5432')
cur=conn.cursor()
curs = conn.cursor()
curs.execute("ROLLBACK")
conn.commit()
cur.execute('SELECT * FROM "project_price_optimization"')

#cur.execute('SELECT * FROM dataoptimize ORDER BY zone, name, brand, mc')
df = cur.fetchall()

## Importing Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy import stats
import pylab
import pickle

df1 = pd.DataFrame(df)

df1=df1.rename( {0 : 'UID'},axis=1)
df1=df1.rename({ 1 : 'NAME'},axis=1)
df1=df1.rename({2 :'ZONE'},axis=1)
df1=df1.rename( {3:'Brand'},axis=1)
df1=df1.rename({4 :'MC'},axis=1)
df1=df1.rename( {5:'Fdate'},axis=1)
df1=df1.rename({6:'quantity'},axis=1)
df1=df1.rename({7:'NSV'},axis=1)
df1=df1.rename({8:'GST_Value'},axis=1)
df1=df1.rename({9:'NSV-GST'},axis=1)
df1=df1.rename({10:'sales_at _cost'},axis=1)
df1=df1.rename({11:'SALES_AT_COST'},axis=1)
df1=df1.rename({12:'MARGIN%'},axis=1)
df1=df1.rename({13:'Gross_Sales'},axis=1)
df1=df1.rename({14:'GrossRGM(P-L)'},axis=1)
df1=df1.rename({15:'Gross_ Margin%(Q/P*100)'},axis=1)
df1=df1.rename({16:'MRP'},axis=1)
df1=df1.rename({17:'price'},axis=1)
df1=df1.rename({18:'DIS'},axis=1)
df1=df1.rename({19:'DIS%'},axis=1)
df1[['quantity','NSV', 'GST_Value', 'NSV-GST', 'sales_at _cost', 'SALES_AT_COST', 'MARGIN%', 'Gross_Sales', 'GrossRGM(P-L)', 'Gross_ Margin%(Q/P*100)', 'MRP', 'price', 'DIS', 'DIS%']] = df1[['quantity','NSV', 'GST_Value', 'NSV-GST', 'sales_at _cost', 'SALES_AT_COST', 'MARGIN%', 'Gross_Sales', 'GrossRGM(P-L)', 'Gross_ Margin%(Q/P*100)', 'MRP', 'price', 'DIS', 'DIS%']].apply(pd.to_numeric)

df1.columns

# checking the Duplicated values present in the datasets
df1[df1.duplicated()]
data = df1.drop_duplicates()

# Checking The null values present in th datasets
data.isnull().sum()
data = data.dropna()
data.shape

data = data.loc[data['MARGIN%'] > 0,:]
data.shape
data.columns


top_10_items  = data['NAME'].value_counts().head(10)
print(top_10_items)

name = input("Enter the product name:")
zone = input("Enter the Zone:")

data1 = data.loc[data['NAME'] == name,:]
data_new = data1.loc[data1['ZONE'] == zone,:]

import numpy as np
from numpy import percentile

data_new.drop(['UID'], axis =1, inplace = True)


##  revenue
# revenue = quantity * price # eq (1)

# revenue = NSU * SP

## profit
# profit = revenue - cost # eq (2)



## revised profit function
# profit = quantity * price - cost # eq (3)

# profit = NSV * SP - cost

def find_optimal_price(data_new):
    import statsmodels.api as sm
    from statsmodels.formula.api import ols  
    # demand curve
    sns.lmplot(x = "price", y = "quantity",data = data_new,fit_reg = True, size = 4)
    # fit OLS model
    model = ols("quantity ~ price", data = data_new).fit()
    # print model summary
    print(model.summary())
    
    fig = plt.figure(figsize=(12,8))
    fig = sm.graphics.plot_partregress_grid(model, fig=fig)
    
    fig = plt.figure(figsize=(12,8))
    fig = sm.graphics.plot_regress_exog(model, "price", fig=fig)
    
    prams = model.params
    prams.Intercept
    prams.price
    
    # plugging regression coefficients
    # quantity = prams.Intercept + prams.price * price # eq (5)
    # the profit function in eq (3) becomes
    # profit = (prams.Intercept + prams.price * price) * price - cost # eq (6)


   # a range of diffferent prices to find the optimum one
    start_price = data_new.price.min() 
    end_price   = data_new.price.max()
    Price  = np.arange(start_price, end_price,0.05)
    Price = list(Price)

   # assuming a fixed cost
    k1   = data_new['sales_at _cost'].div(data_new['quantity'])
    cost = k1.min()
    Profit = []
    Quantity = []
    for i in Price:
       GST = 0.05 * i
       quantity_demanded = prams.Intercept + prams.price * i
       Quantity.append(quantity_demanded)
   
      # profit function
       Profit.append((i - cost - GST) * quantity_demanded)
   # create data frame of price and revenue
    frame = pd.DataFrame({"Price": Price,'Quantity':Quantity,"Profit": Profit })
    
   #plot revenue against price
    plt.plot(frame["Price"], frame["Quantity"])
    plt.plot(frame["Price"], frame["Profit"])
    plt.show()
    
   # price at which revenue is maximum

    ind = np.where(frame['Profit'] == frame['Profit'].max())[0][0]
    values_at_max_profit = frame.iloc[[ind]]
    return values_at_max_profit

# For GH TUR DAL PREM 500g
optimal_price = {}
optimal_price['For GH TUR DAL PREM 500g'] = find_optimal_price(data_new)
optimal_price['For GH TUR DAL PREM 500g'] 


######### Check For Different Items ##########

top_10_items  = superstore['NAME'].value_counts().head(10)
print(top_10_items)

name = input("Enter the product name:")
zone = input("Enter the Zone:")

data = superstore.loc[superstore['NAME'] == name,:] 
data_new = data.loc[data['ZONE'] == zone,:]
print(data)

optimal_price[name] = find_optimal_price(data_new)
optimal_price[name]

#print(data_new['price'].max())

