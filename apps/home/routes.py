import json
import os
import re
from collections import defaultdict
from datetime import datetime
import PyPDF2
import pandas as pd
from flask import render_template, request, url_for, redirect
from flask_login import login_required
from jinja2 import TemplateNotFound
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from apps.home import blueprint


ALLOWED_EXTENSIONS = {'pdf'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_text_from_pdf(pdf_file):
    text = ""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def clean_text(text):
    cleaned_text = re.sub(r'\s+', ' ', text.strip())
    return cleaned_text


def save_text_to_file(text):
    with open("raw_data.txt", "a") as f:
        f.write(text)
        f.write('\n\n')


def process_invoice(cleaned_text):
    # Finding Patterns
    bill_no_pattern = r"BILL NO : (\d+)"
    bill_match = re.search(bill_no_pattern, cleaned_text)
    bill_no = bill_match.group(1) if bill_match else None
    # print(bill_no)

    date_pattern = r"DATE : (\d{2}-\d{2}-\d{4})"
    date_match = re.search(date_pattern, cleaned_text)
    date = date_match.group(1) if date_match else None
    # print(date)

    party_name_pattern = r"PARTY'S NAME :- (\w+ \w+)"
    party_match = re.search(party_name_pattern, cleaned_text)
    party_name = party_match.group(1).strip() if party_match else None
    # print(party_name)

    gst_pattern = r"GST :- ([0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[0-9A-Z]{1}Z[0-9A-Z]{1})"
    gst_match = re.search(gst_pattern, cleaned_text)
    gst = gst_match.group(1).strip() if gst_match else None
    # print(gst)

    gstin_pattern = r"GSTIN :- ([0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[0-9A-Z]{1}Z[0-9A-Z]{1})"
    gstin_match = re.search(gstin_pattern, cleaned_text)
    gstin = gstin_match.group(1).strip() if gst_match else None
    # print(gstin)

    cgst_pattern = r"CGST @ \d+% (\d+)"
    cgst_match = re.search(cgst_pattern, cleaned_text)
    cgst = cgst_match.group(1) if cgst_match else None
    # print(cgst)

    sgst_pattern = r"SGST @ \d+% (\d+)"
    sgst_match = re.search(sgst_pattern, cleaned_text)
    sgst = sgst_match.group(1) if sgst_match else None
    # print(sgst)

    grand_total_pattern = r"Grand Total (\d+,\d+\.\d+)"
    grand_total_match = re.search(grand_total_pattern, cleaned_text)
    grand_total = grand_total_match.group(1) if grand_total_match else None
    # print(grand_total)

    product_pattern = r"\. (.+?) (\d+) (\d+) (\d+) (\d+)"
    product_match = re.findall(product_pattern, cleaned_text)
    product_description = product_match if product_match else None
    # print(product_description)

    # To display df without truncation
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    user_df = pd.DataFrame(product_description, columns=["Product_Name", "HSN_Code", "Qty", "Rate", "Amount"])
    user_df["Bill_No"] = bill_no
    user_df["Bill_Date"] = date
    user_df["Buyer_Name"] = party_name
    user_df["Qty"] = user_df["Qty"].astype(int)
    user_df["Rate"] = user_df["Rate"].astype(int)
    user_df["Amount"] = user_df["Amount"].astype(int)
    temp_product_df = user_df.copy()
    # print(product_df)

    product_name = ", ".join([item[0] for item in product_description])
    hsn_code = ", ".join([item[1] for item in product_description])
    qty = ", ".join([item[2] for item in product_description])
    rate = ", ".join([item[3] for item in product_description])
    amount = ", ".join([item[4] for item in product_description])

    data = {
        "Bill_No": [bill_no],
        "Bill_Date": [date],
        "Seller_GST_No": [gstin],
        "Buyer_Name": [party_name],
        "Buyer_GST_No": [gst],
        "Product_Name": [product_name],
        "HSN_Code": [hsn_code],
        "Qty": [qty],
        "Rate": [rate],
        "Amount": [amount],
        "CGST(9%)": [cgst],
        "SGST(9%)": [sgst],
        "Grand_Total": [grand_total]
    }
    temp_raw_df = pd.DataFrame(data)
    # print(temp_raw_df)
    return temp_product_df, temp_raw_df


def card_values_calculation(df):
    df['Bill_Date'] = pd.to_datetime(df['Bill_Date'], format='%d-%m-%Y')

    order_query = df["Bill_No"].nunique()
    order_result = int(order_query)
    # print(order_result)

    current_month = datetime.now().month
    current_year = datetime.now().year
    current_month_data = df[(df['Bill_Date'].dt.month == current_month) & (df['Bill_Date'].dt.year == current_year)]
    current_month_query = current_month_data["Product_Name"].count()
    current_month_result = int(current_month_query)
    total_sales_query = df["Product_Name"].count()
    total_sales_result = int(total_sales_query)
    # print(total_sales_result)

    current_month_revenue = current_month_data['Amount'].sum()
    revenue_query = df["Amount"].sum()
    revenue_result = int(revenue_query)
    # print(revenue_result)

    total_customers = df['Buyer_Name'].nunique()

    # Find customers who are entirely new in the current month
    previous_month_data = df[(df['Bill_Date'].dt.month < current_month) & (df['Bill_Date'].dt.year == current_year)]
    previous_month_customers = previous_month_data['Buyer_Name'].unique()
    current_month_customers = current_month_data['Buyer_Name'].unique()
    new_customers_current_month = set(current_month_customers) - set(previous_month_customers)

    data = {"order_result": order_result, "total_sales_result": total_sales_result,
            "current_month_result": current_month_result, "revenue_result": revenue_result,
            "current_month_revenue": current_month_revenue, "total_customers": total_customers,
            "new_customers_current_month": len(new_customers_current_month)}
    return data


def apex_chart(df):
    df['Bill_Date'] = pd.to_datetime(df['Bill_Date'], format='%d-%m-%Y')

    # Group by month and year, and count unique orders
    df['Month_Year'] = df['Bill_Date'].dt.strftime('%Y-%m')
    monthly_orders = df.groupby('Month_Year')['Bill_No'].nunique().reset_index()
    monthly_revenue = df.groupby('Month_Year')['Amount'].sum().reset_index()

    # Prepare data for ApexCharts
    months_years = monthly_orders['Month_Year'].tolist()
    orders_count = monthly_orders['Bill_No'].tolist()
    revenue_monthly = monthly_revenue['Amount'].tolist()

    apex_data = {"months_years": months_years, "orders_count": orders_count, "revenue_monthly": revenue_monthly}
    return apex_data


def top_5(df):
    query_result = df.groupby('Buyer_Name').agg({'Amount': 'sum'}).reset_index().nlargest(5, 'Amount')
    return query_result


@blueprint.route('/index')
# @login_required
def index():
    try:
        df = pd.read_csv("generated_data.csv")  # read the CSV file
        if df.empty:  # Check if the DataFrame is empty
            return render_template("home/sample-page.html")
        else:
            data_df = df.to_dict('records')
            data = card_values_calculation(df)
            apex_data = apex_chart(df)
            top_5_data = top_5(df)
            return render_template('home/index.html', data_df=data_df, data=data, apex_data=apex_data,
                                   top_5_data=top_5_data)
    except pd.errors.EmptyDataError:
        return render_template("home/sample-page.html")
    except FileNotFoundError:
        return render_template("home/sample-page.html")


@blueprint.route("/upload", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("home/sample-page.html", message="No file part")
        files = request.files.getlist("file")
        for file in files:
            if file.filename == "":
                return render_template("home/sample-page.html", message="No selected file")
            if file and allowed_file(file.filename):
                # filename = secure_filename(file.filename)

                text = extract_text_from_pdf(file)
                cleaned_text = clean_text(text)
                save_text_to_file(cleaned_text)
                temp_product_df, temp_raw_df = process_invoice(cleaned_text)

                if not temp_product_df.empty and not temp_raw_df.empty:
                    # Save product DataFrame to CSV
                    if not os.path.isfile("product_data.csv") or os.stat("product_data.csv").st_size == 0:
                        temp_product_df.to_csv("product_data.csv", index=False)  # Write DataFrame with headers
                    else:
                        temp_product_df.to_csv("product_data.csv", mode='a', header=False, index=False)

                    # Save summary DataFrame to CSV
                    if not os.path.isfile("summary_data.csv") or os.stat("summary_data.csv").st_size == 0:
                        temp_raw_df.to_csv("summary_data.csv", index=False)
                    else:
                        temp_raw_df.to_csv("summary_data.csv", mode='a', header=False, index=False)
                    # message = f"File '{file.filename}' uploaded successfully"
                    # return render_template("home/index.html", message=message, data=card_values_calculation())
                else:
                    message = f"Failed to extract data from the invoice '{file.filename}'"
                    return render_template("home/index.html", message=message)
            else:
                message = f"Error: Invalid file type. Only PDF files are allowed."
                return render_template("home/sample-page.html", message=message)
        return redirect(url_for('home_blueprint.index'))
    return render_template("home/sample-page.html")


def various_analysis(df):
    # Convert 'Bill_Date' to datetime
    df['Bill_Date'] = pd.to_datetime(df['Bill_Date'], format='%d-%m-%Y')
    df['Month_Year'] = df['Bill_Date'].dt.strftime('%Y-%m')

    # Sort DataFrame by 'Bill_Date' to ensure accurate calculations
    df.sort_values(by='Bill_Date', inplace=True)

    # Calculate time between consecutive purchases for each customer
    df['Time_Between_Purchases'] = df.groupby('Buyer_Name')['Bill_Date'].diff()

    # Calculate average time between purchases for each customer
    avg_time_between_purchases = df.groupby('Buyer_Name')['Time_Between_Purchases'].mean()

    # Calculate frequency of purchases for each customer
    purchase_frequency = df.groupby('Buyer_Name').size()

    # Calculate recency (time since last purchase) for each customer
    latest_purchase_date = df.groupby('Buyer_Name')['Bill_Date'].max()
    recency = pd.Timestamp.now() - latest_purchase_date

    # Calculate monetary value (total spending) for each customer
    total_spending = df.groupby('Buyer_Name')['Amount'].sum()

    # Calculate basket size (average quantity per transaction) for each customer
    basket_size = df.groupby(['Buyer_Name', 'Bill_No'])['Qty'].sum().groupby('Buyer_Name').mean()

    # Total quantity of each product
    total_qty_sold = df.groupby('Product_Name')['Qty'].sum().reset_index()

    # Total revenue generated by each product
    total_revenue = df.groupby('Product_Name')['Amount'].sum().reset_index()

    # Merge the total quantity and total revenue DataFrames
    analysis_df = pd.merge(total_qty_sold, total_revenue, on='Product_Name')

    # Calculate price per unit for each product
    analysis_df['Price_Per_Unit'] = analysis_df['Amount'] / analysis_df['Qty']

    # customer behaviour
    customer_purchase_summary = df.groupby('Buyer_Name').agg({
        'Qty': 'sum',
        'Rate': ['sum', 'mean'],
        'Product_Name': 'count'
    })

    customer_purchase_summary.columns = ['Total_Quantity', 'Total_Revenue', 'Average_Rate', 'Total_Products_Purchased']
    # Convert customer_purchase_summary DataFrame to dictionary
    customer_purchase_summary_dict = customer_purchase_summary.to_dict()

    # Combine loyalty metrics into a DataFrame
    loyalty_metrics = pd.DataFrame({
        # 'Average_Time_Between_Purchases': avg_time_between_purchases,
        # 'Purchase_Frequency': purchase_frequency,
        # 'Recency': recency,
        # 'Total_Spending': total_spending,
        # 'Basket_Size': basket_size,
        "customers": avg_time_between_purchases.index,
        "avg_time": avg_time_between_purchases.values,
        "frequency": purchase_frequency.values,
        "recency": recency.dt.days.values,
        "spending": total_spending.values,
        "basket": basket_size.values
    })

    data = {
        "customers": loyalty_metrics['customers'].tolist(),
        "product_name": analysis_df['Product_Name'].tolist(),
        "total_qty": analysis_df['Qty'].tolist(),
        "total_revenue": analysis_df['Amount'].tolist(),
        "average_time": [str(td) for td in loyalty_metrics['avg_time']],
        "purchase_frequency": loyalty_metrics['frequency'].tolist(),
        "recency": loyalty_metrics['recency'].tolist(),
        "total_spending": loyalty_metrics['spending'].tolist(),
        "basket_size": loyalty_metrics['basket'].tolist(),
        "price_per_unit": analysis_df['Price_Per_Unit'].tolist(),
        "date": df['Bill_Date'].dt.strftime('%Y-%m').tolist(),
        "customer_purchase_summary": customer_purchase_summary_dict,
        "rate": df['Rate'].tolist(),
        "qty": df['Qty'].tolist()
    }
    return data


def sales_trend(df):
    # Convert 'Bill_Date' column to datetime format
    df['Bill_Date'] = pd.to_datetime(df['Bill_Date'], format='%d-%m-%Y')

    # Group data by 'Product_Name' and 'Buyer_Name' and aggregate quantities and amounts
    grouped_data = df.groupby(['Product_Name', 'Buyer_Name']).agg({
        'Qty': 'sum',
        'Amount': 'sum',
        'Bill_Date': lambda x: pd.Series(x).min(),  # Get the earliest date for each group
    }).reset_index()

    # Create a DataFrame to store all aggregated data
    all_data = pd.DataFrame(columns=['Product_Name', 'Buyer_Name', 'Bill_Date', 'Qty', 'Amount'])

    # Append all data to the all_data DataFrame
    for index, row in grouped_data.iterrows():
        product = row['Product_Name']
        buyer = row['Buyer_Name']
        quantity = row['Qty']
        amount = row['Amount']
        first_purchase_date = row['Bill_Date']

        product_data = df[(df['Product_Name'] == product) & (df['Buyer_Name'] == buyer)]
        all_data = pd.concat([all_data, product_data])

    # Time Series Analysis for Quantity
    quantity_ts = all_data.groupby('Bill_Date')['Qty'].sum()
    quantity_decomposition = seasonal_decompose(quantity_ts, model='additive',
                                                period=12)  # Assuming a yearly seasonality
    quantity_trend = quantity_decomposition.trend
    quantity_seasonal = quantity_decomposition.seasonal

    # Time Series Analysis for Amount
    amount_ts = all_data.groupby('Bill_Date')['Amount'].sum()
    amount_decomposition = seasonal_decompose(amount_ts, model='additive', period=12)  # Assuming a yearly seasonality
    amount_trend = amount_decomposition.trend
    amount_seasonal = amount_decomposition.seasonal

    # Prepare data for JavaScript
    quantity_ts_json = quantity_ts.reset_index().to_json(orient='records')
    amount_ts_json = amount_ts.reset_index().to_json(orient='records')
    categories_json = json.dumps(list(quantity_ts.index.astype(str)))

    # data = {"quantity_ts_json": quantity_ts_json, "amount_ts_json":amount_ts_json, "categories_json": categories_json}
    return quantity_ts_json, amount_ts_json, categories_json


def calculate_monthly_growth_with_prediction(df):
    # Convert 'Bill_Date' to datetime
    df['Bill_Date'] = pd.to_datetime(df['Bill_Date'], format='%d-%m-%Y')
    df['Month_Year'] = df['Bill_Date'].dt.strftime('%Y-%m')

    # Sort DataFrame by 'Bill_Date' to ensure accurate calculations
    df.sort_values(by='Bill_Date', inplace=True)

    # Initialize dictionary to store monthly growth for each product
    monthly_growth = defaultdict(dict)
    growth_predictions = {}

    for _, row in df.iterrows():
        product_name = row['Product_Name']
        qty = row['Qty']
        bill_date = row['Bill_Date']

        # Extract year and month from the bill date
        year_month = bill_date.strftime('%Y-%m')

        # Initialize product data if not already present
        if product_name not in monthly_growth:
            monthly_growth[product_name] = defaultdict(int)

        # Add quantity to the corresponding month for the product
        monthly_growth[product_name][year_month] += qty

    # Calculate monthly growth for each product and predict growth for the next month
    for product, monthly_data in monthly_growth.items():
        months = sorted(monthly_data.keys())
        X_train = np.array([[int(month.split('-')[1])] for month in months])
        y_train = np.array([monthly_data[month] for month in months])

        # Initialize and train Random Forest regressor
        rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_regressor.fit(X_train, y_train)

        # Predict growth for the next month
        next_month = int(months[-1].split('-')[1]) + 1
        next_month_growth = rf_regressor.predict([[next_month]])[0]
        growth_predictions[product] = next_month_growth

        # Calculate growth for each month
        for i in range(1, len(months)):
            current_month = months[i]
            previous_month = months[i - 1]
            current_qty = monthly_data[current_month]
            previous_qty = monthly_data[previous_month]
            growth = (current_qty - previous_qty) / previous_qty * 100 if previous_qty != 0 else 0
            monthly_growth[product][current_month] = growth

    return monthly_growth, growth_predictions


@blueprint.route('/chart-apex')
def smart_analysis():
    try:
        df = pd.read_csv("generated_data.csv")  # read the CSV file
        if df.empty:  # Check if the DataFrame is empty
            return render_template("home/sample-page.html")
        else:
            data = various_analysis(df)
            monthly_sales_data, growth_predictions = calculate_monthly_growth_with_prediction(df)
            # quantity_data, amount_data, categories_json = sales_trend(df)
            return render_template('home/chart-apex.html', data=data, monthly_sales_data=monthly_sales_data,
                                   growth_predictions=growth_predictions)
    except pd.errors.EmptyDataError:
        return render_template("home/sample-page.html")
    except FileNotFoundError:
        return render_template("home/sample-page.html")


@blueprint.route('/<template>')
@login_required
def route_template(template):
    try:

        if not template.endswith('.html'):
            template += '.html'

        # Detect the current page
        segment = get_segment(request)

        # Serve the file (if exists) from app/templates/home/FILE.html
        return render_template("home/" + template, segment=segment)

    except TemplateNotFound:
        return render_template('home/page-404.html'), 404

    except:
        return render_template('home/page-500.html'), 500


# Helper - Extract current page name from request
def get_segment(request):
    try:

        segment = request.path.split('/')[-1]

        if segment == '':
            segment = 'index'

        return segment

    except:
        return None
