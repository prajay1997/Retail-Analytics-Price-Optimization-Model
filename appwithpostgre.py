import streamlit as st
import pickle
import pandas as pd
import numpy as np
import psycopg2
conn=psycopg2.connect(**st.secrets["postgres"])
cur=conn.cursor()
curs = conn.cursor()
curs.execute("ROLLBACK")
conn.commit()
cur.execute('SELECT * FROM "project_price_optimization"')

#cur.execute('SELECT * FROM dataoptimize ORDER BY zone, name, brand, mc')
df = cur.fetchall()

if (len(df) == 0):
    st.write('No records found in DB')
else:
    import pandas as pd

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

    st.title('Price Optimization')

    Unique_Products =pickle.load(open('Unique_Products.pkl','rb'))
    Zone = pickle.load(open('Zone.pkl','rb'))
    model = pickle.load(open('model.pkl','rb'))


    Selected_Product_Name = st.selectbox(
        'Select Product Name',
         (Unique_Products.values))

    Selected_Zone = st.selectbox(
        'Select Zone',
         (Zone.values))

    data = data.loc[data['NAME'] == Selected_Product_Name,:]
    data_new = data.loc[data['ZONE'] == Selected_Zone,:]
    values_at_max_profit = 0
    def find_optimal_price(data_new):
        import statsmodels.api as sm
        from statsmodels.formula.api import ols
        # demand curve
        # sns.lmplot(x = "price", y = "quantity",data = data_new,fit_reg = True, size = 4)
        # fit OLS model
        model = ols("quantity ~ price", data=data_new).fit()
        # print model summary
        print(model.summary())
        prams = model.params

        # plugging regression coefficients
        # quantity = prams.Intercept + prams.price * price # eq (5)
        # the profit function in eq (3) becomes
        # profit = (prams.Intercept + prams.price * price) * price - cost # eq (6)

        # a range of diffferent prices to find the optimum one
        start_price = data_new.price.min()
        end_price = data_new.price.max()
        Price = np.arange(start_price, end_price, 0.05)
        Price = list(Price)

        # assuming a fixed cost
        k1 = data_new['NSV'].div(data_new['quantity'])
        cost = k1.min()
        Revenue = []
        for i in Price:
            quantity_demanded = prams.Intercept + prams.price * i

            # profit function
            Revenue.append((i - cost) * quantity_demanded)
        # create data frame of price and revenue
        profit = pd.DataFrame({"Price": Price, "Revenue": Revenue})

        # plot revenue against price
        #plt.plot(profit["Price"], profit["Revenue"])

        # price at which revenue is maximum

        ind = np.where(profit['Revenue'] == profit['Revenue'].max())[0][0]
        values_at_max_profit = profit.iloc[[ind]]
        return values_at_max_profit


    #optimal_price = {}
    #optimal_price[Selected_Product_Name] = find_optimal_price(data_new)
    #optimal_price[Selected_Product_Name]

    if st.button('Predict Optimized Price'):
        values_at_max_profit = find_optimal_price(data_new)
        st.write('Optimized Price of the Product', values_at_max_profit )

