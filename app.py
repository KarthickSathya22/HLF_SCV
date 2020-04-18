import tabula
import dateutil
import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

#Loading a model:
model = pickle.load(open('model_scv_iso.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/back',methods=['POST','GET'])
def back():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    predict_request = []
    res = []
    
    status = request.form["martial_status"]
    married = {2750:"Married",2751:"Un Married"}
    predict_request.append(status)
    res.append(married.get(int(status)))
    
    dep = request.form["dependants"]
    predict_request.append(dep)
    res.append(dep)
    
    resi = request.form["residence"]
    residence = {2755:"Own",2756:"Rent"}
    predict_request.append(resi)
    res.append(residence.get(int(resi)))
    
    year = request.form["staying_year"]
    predict_request.append(year)
    res.append(year)
    
    age = request.form["age"]
    predict_request.append(age)
    res.append(age)
    
    indus = request.form["industrytype"]
    ind_cat = {1782:"Salaried",1783:"Self Employeed",603:"Agriculture",
     604:"Passenger Transportation",605:"Construction",875:"Infrastructure",
     876:"Cement",877:"Oil and Gas",878:"Government Contract",879:"Others",658:"Mine"}
    predict_request.append(indus)
    res.append(ind_cat.get(int(indus)))
    
    profile = request.form["profile"]
    pro_cat = {2692:"Captive Class",2693:"Retail Class",2694:"Strategy Class"}
    predict_request.append(profile)
    res.append(pro_cat.get(int(profile)))
    
    segment = request.form["segment"]
    seg_cat = {2695:"First Time Buyer",
            2696:"First Time Buyer Plus",
            2697:"Medium Fleet Operators",
            2698:"Small Fleet Operators"}
    predict_request.append(segment)
    res.append(seg_cat.get(int(segment)))
    
    market = request.form["market"]
    market_cat = {2729:"Completely Market",2730:"Attached to Transporter"}
    predict_request.append(market)
    res.append(market_cat.get(int(market)))
    
    years = request.form["tot_years"]
    predict_request.append(years)
    res.append(years)
    
    cat = request.form["productcat"]
    prod_cat = {1784:"Loan Against Property",
            926:"Car",
            912:"Multi Utility Vehicle",
            945:"Vikram",
            1402:"Tractor",
            1373:"Used Vehicles",
            1672:"Tipper",
            1664:"Farm Equipment",
            1541:"Two Wheeler",
            634:"Intermediate Commercial Vehicle",
            527:"Heavy Commercial Vehicle",
            528:"Construction Eqquipments",
            529:"Three Wheelers",
            530:"Light Commercial Vehicle",
            531:"Small Commercial Vehicle",
            738:"Medium Commercial Vehicle",
            783:"Busses"}
    predict_request.append(cat)
    res.append(prod_cat.get(int(cat)))
    
    tenure = request.form["tenure"]
    predict_request.append(tenure)
    res.append(tenure)
    
    instal = request.form["instalcount"]
    predict_request.append(instal)
    res.append(instal)
    
    chasasset = request.form["chasasset"]
    predict_request.append(chasasset)
    res.append(chasasset)
    
    chasinitial = request.form["chasinitial"]
    predict_request.append(chasinitial)
    res.append(chasinitial)
    
    chasfin = int(chasasset) - int(chasinitial)
    predict_request.append(chasfin)
    res.append(chasfin)
    
    fininter = request.form["finaninterest"]
    predict_request.append(fininter)
    res.append(fininter)
    
    interestamount = (int(chasfin)*(int(tenure)/12)*(float(fininter)))/100
    emi = (int(chasfin)+int(interestamount))/int(tenure)
    predict_request.append(int(emi))
    res.append(int(emi))
    
    gross_loan = request.form["gross_loan"]
    predict_request.append(gross_loan)
    res.append(gross_loan)
    
    income = request.form["totincome"]
    predict_request.append(income)
    res.append(income)
    
    expense = request.form["totexpense"]
    predict_request.append(expense)
    res.append(expense)
    
    surplus = int(income) - int(expense)
    predict_request.append(surplus)
    res.append(surplus)
    
    veh_age = request.form["vehicle_age"]
    predict_request.append(veh_age)
    res.append(veh_age)
    
    brand = request.form["brand"]
    brand_type = {746:"Mahindra",
                  747:"Piaggio",
                  564:"Ashok Leyland",
                 1437:"Atul Auto",
                  821:"Maruti Suzuki",
                  816:"Bajaj Auto",
                  908:"Atul Shakti",
                  742:"Force Motors",
                  723:"Eicher Motors",
                 1654:"Continental Engines Ltd",
                 1341:"Hyundai Motors",
                 1491:"Renault",
                 2035:"Lohia Industries",
                 1342:"Ford India Ltd",
                 1523:"Maruthi",
                 1415:"Nissan",
                 1407:"API Motors Ltd",
                 1330:"JSA",
                 1420:"Toyota Motors",
                  935:"TVS Motors",
                 2034:"Pasupathi Vehicles Ltd",
                  724:"Swaraj Mazda Ltd",
                 1440:"Scooter India Ltd",
                  914:"Toyota Kirloskar Motors",
                 1404:"Chevrolet",
                 1391:"Volswagen",
                 1360:"Honda Motors"}
    predict_request.append(brand)
    res.append(brand_type.get(int(brand)))
    
    
    #Uploading file:
    file = request.files['file']
    filename = file.filename
    extn = filename.split('.')[-1]   
    destination = file
  
    #Checking for extension of file: 
    if (extn.casefold() == 'pdf'):
        tables = tabula.read_pdf(destination,pages='all')
        
        #Combining all tables:
        all_table = []
        for i in range(len(tables)):
            all_table.extend(tables[i].values.tolist())
            
        #Creating dataframe:
        df = pd.DataFrame(all_table)
            
        #Reset the Index:
        df.reset_index(drop=True, inplace=True)
        
        #Parsing fields:
        df[0] = df[0].astype(str)
        df[6] = df[6].astype(str)        
        
        #Checking for extra cloumns:
        if df.shape[1]>7:
            df[7:] = df[7:].astype(str)
        
        #Converting into list:
        list_rows = df.values.tolist()
        
        #Removing unwanted rows:
        rows = []
        for i in range(len(list_rows)):
            if (list_rows[i][0] == 'nan'):
                pass
            else:
                rows.append(list_rows[i])
                
        new = pd.DataFrame(rows)
        
        list_rows = new.values.tolist()
        
        #Calculating closing price: 
        rows = []
        pos = 6
        if (len(list_rows[0])>7):
            for i in range(len(list_rows)):
                if ((list_rows[i][pos] == 'nan') | (list_rows[i][pos] == 'None')):
                    if (list_rows[i][pos+1] != 'None'):
                        list_rows[i][pos] = list_rows[i][pos+1]
                    else:
                        list_rows[i][pos]  = list_rows[i][pos-1]
                        list_rows[i][pos-1] = np.nan
                del list_rows[i][pos+1:]
                rows.append(list_rows[i])
        else:
            rows = list_rows
        
        final = pd.DataFrame(rows)
        
        #Adding Features Names:
        final.columns = ['Date', 'Narration', 'Chq./Ref.No.', 'Value Dt', 'Withdrawal Amt.','Deposit Amt.', 'Closing Balance']
        
        #Reset the Index:
        final.reset_index(drop=True, inplace=True)
        
        #Parsing date:
        final['Date'] = final['Date'].apply(dateutil.parser.parse, dayfirst=True)
        final.head()
        
        #Paring prices:
        final['Closing Balance'] = final['Closing Balance'].astype(str)
        col = ['Closing Balance']
        for i in col:
            val = []
            for j in final[i]:
                val.append(''.join(j.split(',')))
            final[i] = val
            
        #Converting price:
        col = ['Closing Balance']
        for i in col:
            final[i] = pd.to_numeric(final[i],errors='coerce')
            
        #Group by operation to close price:
        group = final.groupby(pd.Grouper(key='Date',freq='1M'))
        
        #Filtering close balance per month:
        balance_month = []
        for i in group:
            a = i[1]
            balance_month.append(a['Closing Balance'].iloc[-1])
        
        #Returnig a result:
        clobal =  (np.average(balance_month))
    
    if (extn.casefold() == 'xls'):
        #Loading dataset:
        df = pd.read_excel(destination)
        
        #Fetching transactions only:
        row_no = 0
        for i in df.iloc[:,0]:
            if i == 'Date':
                df = df.iloc[row_no:]
                break
            row_no = row_no+1
        
        #Set a features name:
        df.columns = ['Date', 'Narration', 'Chq./Ref.No.', 'Value Dt', 'Withdrawal Amt.','Deposit Amt.', 'Closing Balance']
        
        #Reset the Index:
        df.reset_index(drop=True, inplace=True)
        
        #Dropping first two records:
        df.drop([0,1],axis=0,inplace=True)
        
        #Reset the Index:
        df.reset_index(drop=True, inplace=True)
        
        row_no = 0
        for i in df['Date']:
            if len(str(i)) != 8:
                df = df.iloc[:row_no]
                break
            row_no = row_no + 1
            
        # Parsing date:
        df['Date'] = df['Date'].apply(dateutil.parser.parse, dayfirst=True)
        table = df
        
        #Group by operation to find opening and close price:
        group = table.groupby(pd.Grouper(key='Date',freq='1M'))
        
        #Filtering open and close balance per month:
        balance_month = []
        for i in group:
            a = i[1]
            balance_month.append(a['Closing Balance'].iloc[-1])
        
        clobal = (np.average(balance_month))
   
    predict_request.append("{:.2f}".format(clobal))
    res.append("{:.2f}".format(clobal))
    
    gender_dict = {'M':[0,1],'F':[1,0]}
    cate = request.form["gender"]
    if cate == 'M':
        res.append('Male')
    else:
        res.append('Female')
    predict_request.extend(gender_dict.get(cate))
    predict_request = list(map(float,predict_request))
    predict_request = np.array(predict_request)
    prediction = model.predict_proba([predict_request])[0][-1]
    output = int((1 - prediction)*100)
    if output < 50:
        condition = 'Risky'
    if output >= 50 and output <= 69:
        condition = 'Barely Acceptable'
    if output >= 70 and output <=89:
        condition = 'Medium'
    if output >= 90 and output <= 99:
        condition = 'Good'
    if output == 100:
        condition = 'Superior'
        
    return render_template('resultpage.html', prediction_text=output,data=res,status=condition)

if __name__ == "__main__":
    app.run(debug=True)
