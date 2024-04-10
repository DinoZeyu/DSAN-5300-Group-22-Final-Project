import pandas as pd
from sklearn.preprocessing import LabelEncoder

## Read file 
df1 = pd.read_csv('Data/listings_detail.csv')

## Drop unnecessary features by hand
df1 = df1.drop(columns = ['id',"host_id",'name','listing_url', 'scrape_id', 'last_scraped', 'source','description', 'picture_url','host_url', 'host_name',
                          'host_thumbnail_url', 'host_picture_url','host_neighbourhood','host_listings_count','host_verifications',
                          'neighbourhood','neighbourhood_cleansed', 'neighbourhood_group_cleansed', 'latitude','longitude','bathrooms',
                          'bedrooms','amenities','minimum_minimum_nights','maximum_minimum_nights', 'minimum_maximum_nights',
                          'maximum_maximum_nights', 'minimum_nights_avg_ntm','maximum_nights_avg_ntm',"calendar_updated",'calendar_last_scraped',
                          'number_of_reviews_ltm', 'number_of_reviews_l30d','license','calculated_host_listings_count','calculated_host_listings_count_entire_homes',
                          'calculated_host_listings_count_private_rooms','calculated_host_listings_count_shared_rooms','neighborhood_overview', 'host_about'])

## Remove NA values
df1.dropna(subset=df1.columns.difference(['review_scores_rating']), inplace=True)


## Assign Labels for Popularity as multiple classes

quintiles = [0, 0.25, 0.5, 0.75, 1.0]
quintile_values = df1['number_of_reviews'].quantile(quintiles)
bins = [-1, quintile_values[0.25], quintile_values[0.5], quintile_values[0.75], float('inf')]

labels = ['low popularity', 'moderate popularity', 'high popularity', 'very high popularity']

df1['popularity'] = pd.cut(df1['number_of_reviews'], bins=bins, labels=labels, include_lowest=True)

## Remove $ sign from price 
df1['price'] = df1['price'].str.replace('$', '')
df1['price'] = df1['price'].str.replace(',', '').astype('float')

## Transfer % to float
df1['host_response_rate'] = df1['host_response_rate'].str.replace('%','').astype('float') / 100
df1['host_acceptance_rate'] = df1['host_acceptance_rate'].str.replace('%','').astype('float') / 100

## Extract number of bathrooms
df1['bathrooms_text'] = df1['bathrooms_text'].str.extract('(\d+)').astype('float')

# host_is_local
def is_local(location):
  return 1 if location == 'Bangkok, Thailand' else 0

df1['host_is_local'] = df1['host_location'].apply(is_local)


# Remove unneeded columns
df1.drop(columns = (['host_location', 'property_type', 'number_of_reviews']), inplace=True)


## Create encoder method
def encode_columns(df, columns):
    encoder = LabelEncoder()
    for column in columns:
        df[column] = encoder.fit_transform(df[column])
    return df

# List of columns you want to encode
columns_to_encode = [
    'popularity',
    'host_is_superhost',
    'host_has_profile_pic',
    'host_identity_verified',
    'has_availability',
    'instant_bookable',
    'host_response_time',
    'room_type'
]

df1 = encode_columns(df1, columns_to_encode)

# Calculate the difference in years
import datetime as dt
df1['host_since'] = pd.to_datetime(df1['host_since'])
current_date = dt.datetime.now().date()
df1['host_years'] = df1['host_since'].apply(lambda x: current_date.year - x.year - ((current_date.month, current_date.day) < (x.month, x.day)))
df1.drop(columns=['host_since'],inplace=True)

# Calculate reviews years range
df1['first_review'] = pd.to_datetime(df1['first_review'])
df1['last_review'] = pd.to_datetime(df1['last_review'])
df1['review_years_range']= round(((df1['last_review'] - df1['first_review']) / pd.Timedelta(days=365.25)).astype(float),2)
df1.drop(columns=['first_review','last_review'],inplace=True)

## Store the clean data for using R 
df1.to_csv('Data/cleaned.csv', index=False)
