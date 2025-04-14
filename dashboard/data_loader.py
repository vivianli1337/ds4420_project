import pandas as pd

# list of valid clothing items to keep from the dataset
clothing_items = [
    'Tunic', 'Tank Top', 'Leggings', 'Onesie', 'Jacket', 'Trousers', 'Jeans',
    'Pajamas', 'Trench Coat', 'Poncho', 'Romper', 'T-shirt', 'Shorts',
    'Blazer', 'Hoodie', 'Sweater', 'Blouse', 'Swimsuit', 'Kimono', 'Cardigan',
    'Dress', 'Camisole', 'Flannel Shirt', 'Polo Shirt', 'Overalls', 'Coat',
    'Vest', 'Jumpsuit', 'Raincoat', 'Skirt', 'Pants'
]

def load_and_clean_data(csv="retail_sales.csv"):
    # load data from csv
    df = pd.read_csv(csv)
    
    # convert date column to datetime format
    df['date'] = pd.to_datetime(df['date'])
    
    # remove rows with missing sales amounts
    df = df.dropna(subset=['amount_usd'])
    
    # create a new column for month using timestamp
    df['month'] = df['date'].dt.to_period('M').dt.to_timestamp()
    
    # keep only rows where item is in the list of clothing items
    df = df[df['item'].isin(clothing_items)].copy()
    
    return df

def prepare_monthly_sales(df):
    # group data by item and month, summing sales and counting units sold
    monthly_sales = df.groupby(['item', 'month']).agg(
        Total_Sales=('amount_usd', 'sum'),
        Units_Sold=('item', 'count')
    ).reset_index()
    
    return monthly_sales
