#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import All Libraries.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from urllib.request import urlopen
from bs4 import BeautifulSoup
import requests
from pandas.io.json import json_normalize

get_ipython().system('conda install -c conda-forge folium=0.5.0')
import folium

print('All necessary libraries Imported')


# In[3]:


# Web Scraping and creating dataframe.
url = "https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M"
html = urlopen(url) # Convert url to html
soup = BeautifulSoup(html) # Create Soup


# In[4]:


# Extract all tables form the page. Visit page for a clear idea.
tables = soup.find_all('table') 

# Extract only First Table as only interested in Postcode, Borough and Neighbourhood Table.
table = tables[0] 

# Find all rows in the table.
rows = table.find_all('tr') 

# Create list of cleaned data
list_rows = []
for row in rows:
    cols = row.find_all('td') # Find column data in each row.
    clean_row=[]
    for col in cols:
        data = col.find(text=True).strip() # Find only text from each column data and remove new-line characters.
        clean_row.append(data)
    list_rows.append(clean_row)

 # Create data frame from cleaned data
df_postcode = pd.DataFrame(list_rows[1:], columns=['Postcode','Borough','Neighborhood'])


# In[5]:


# Convert all data to strings
df_postcode = df_postcode.astype('str')


# In[6]:


# Remove all rows where Borough is not assigned
df_postcode = df_postcode[df_postcode['Borough'] != 'Not assigned']


# In[7]:


# If Neighborhood is Not-Assigned, assign the same value as Borough
for x in df_postcode['Neighborhood']:
    if x == 'Not assigned\n':
        x = df_postcode['Borough']


# In[8]:


# Groupby Postcode + Borough & join Neighborhood with ','
df_postcode_final = df_postcode.groupby(['Postcode','Borough'])['Neighborhood'].apply(lambda x: ','.join(x)).reset_index()


# In[9]:


# Check results
print('Shape of final postcode data is {}'.format(df_postcode_final.shape))
df_postcode_final.head()


# In[10]:


# Read Geo-Spatial Data and rename colun name
df_geo_data = pd.read_csv('http://cocl.us/Geospatial_data')
df_geo_data.rename(columns={'Postal Code':'Postcode'}, inplace=True)

print('Shape of df_geo_data: {}'.format(df_geo_data.shape))
print('Head of df_geo_data: \n{}'.format(df_geo_data.head()))


# In[11]:


# Merge df_postcode_data and df_geo_code based on postal code.
df_data = pd.merge(df_postcode_final, df_geo_data, on='Postcode', how='inner')

print('Shape of df_data = No of unique neighborhood/postcode: {}'.format(df_data.shape))
df_data.head()


# In[12]:


#!conda install -c conda-forge geopy --yes 
from geopy.geocoders import Nominatim
# Retrieve Lati and Long of Toronto for Map creation.
address = 'Toronto, Canada'

geolocator = Nominatim(user_agent="explorer")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The geograpical coordinate of {} are {}, {}.'.format(address, latitude, longitude))


# In[13]:


# create map of Toronto using latitude and longitude values
map_toronto = folium.Map(location=[latitude, longitude], zoom_start=10)

# add markers to map
for lat, lng, borough, neighborhood in zip(df_data['Latitude'], df_data['Longitude'], df_data['Borough'], df_data['Neighborhood']):
    label = '{}, {}'.format(neighborhood, borough)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=4,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map_toronto)


# In[14]:


map_toronto


# In[15]:


# Setup Foursquare Credentials
CLIENT_ID = 'YWLNZ2TWMJ4KDQGPEDQSEKMZU2L1SHPZXIMK42OCON21P5R5' # your Foursquare ID
CLIENT_SECRET = 'QOZSNPHZWO2RS2I01O4C1RF2G0SAVCEN0TT1BGBXMU3AMJUP' # your Foursquare Secret
VERSION = '20190901' # Foursquare API version

RADIUS = 500 # Radius of search in meters
LIMIT = 100 # Limit the count of search


# In[16]:


def getNearbyVenues(neighborhood, lati, longi):
    
    venues_list=[]
    for name, LATI, LONGI in zip(neighborhood, lati, longi):
        print(name)
        
        # Setup url for API call
        url = 'https://api.foursquare.com/v2/venues/search?client_id={}&client_secret={}&ll={},{}&v={}&radius={}&limit={}'                                             .format(CLIENT_ID, CLIENT_SECRET, LATI, LONGI, VERSION, RADIUS, LIMIT)
            
       # API call to get the json file
        results = requests.get(url).json()
        
        # Extract all the necessary categories from the venue.
        venue_details = results['response']['venues']

        category_details = []
        for venue in venue_details:
            category_details.append(venue['categories'])

        for detail in category_details:
            if len(detail) != 0:
                venues_list.append((name,detail[0]['name']))


    nearby_venues = pd.DataFrame(venues_list)
    nearby_venues.columns = ['Neighborhood', 'Venue Category']
    
    return(nearby_venues)


# In[17]:


df_toronto_venues = getNearbyVenues(neighborhood=df_data['Neighborhood'], lati=df_data['Latitude'], longi=df_data['Longitude'])


# In[18]:


print('Shape of df_toronto_venues: {}'.format(df_toronto_venues.shape))
print('Total no of unique neighborhoods in df_toronto_venues: {}'.format(len(np.unique(df_toronto_venues['Neighborhood']))))
df_toronto_venues.head()


# In[19]:


# Drop duplicate values
df_toronto_venues = df_toronto_venues.drop_duplicates(['Neighborhood','Venue Category'], keep='first')

print('Shape of df_toronto_venues: {}'.format(df_toronto_venues.shape))
print('Total no of Unique Category in df_toronto_venues: {}'.format(len(np.unique(df_toronto_venues['Venue Category']))))
df_toronto_venues.head()


# In[20]:


# Apply One Hot Encoding on df_toronto_venues
venue_onehot = pd.get_dummies(df_toronto_venues[['Venue Category']],prefix='',prefix_sep='')

print('Shape of venue_onehot: {}'.format(venue_onehot.shape))
print('Total no of Categories in venue_onehot: {}'.format(len(venue_onehot.columns)))
venue_onehot.head()


# In[21]:


# Add neighborhood to venue_onehot and move it to 1st column for easy understanding
venue_onehot['Neighborhood'] = df_toronto_venues['Neighborhood']
new_col_seq = [venue_onehot.columns[-1]] + list(venue_onehot.columns[:-1])
venue_onehot = venue_onehot[new_col_seq]


print('Shape of venue_onehot: {}'.format(venue_onehot.shape))
print('Total no of Categories in venue_onehot: {}'.format(len(venue_onehot.columns)-1))
venue_onehot.head()


# In[22]:


# Group_by venue_onehot based on neighborhood to match df_toronto for joining. Here mean is considered.
neighborhood_onehot = venue_onehot.groupby('Neighborhood').mean().reset_index()

print('Total neighborhoods = Total Postcodes:{}'.format(neighborhood_onehot.shape[0]))
print('Total no of diff venue categories used: {}'.format(neighborhood_onehot.shape[1]-1))
neighborhood_onehot.head()


# In[23]:


# Create an Empty dataframe which shows top n venue categories in a neighborhood.
num_top_venues = 10

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['Neighborhood']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
neighborhood_top_venues = pd.DataFrame(columns=columns)
neighborhood_top_venues['Neighborhood'] = neighborhood_onehot['Neighborhood']

neighborhood_top_venues.head()


# In[24]:


# Add values to the above dataframe.
def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)    
    return row_categories_sorted.index.values[0:num_top_venues]

for row in np.arange(neighborhood_onehot.shape[0]):
    neighborhood_top_venues.iloc[row, 1:] = return_most_common_venues(neighborhood_onehot.iloc[row, :], num_top_venues)
    
neighborhood_top_venues


# In[89]:


# Apply Kmeans clustering
from sklearn.cluster import KMeans
neighborhood_model_data = neighborhood_onehot.drop(['Neighborhood'], axis=1)

k = 5
kmeans = KMeans(n_clusters=k, random_state=0).fit(neighborhood_model_data)
len(kmeans.labels_)


# In[90]:


# Is done to avoid rerun of complete code for any issues.
neighborhood_top_clustered_venues = neighborhood_top_venues.copy(deep=True)


# In[91]:


# Add labels to neighborhood_top_clustered_venues
neighborhood_top_clustered_venues.insert(1, 'Cluster Labels', kmeans.labels_)
neighborhood_top_clustered_venues


# In[92]:


# Join df_data and neighborhood_top_clustered_venues on Neighborhood for final o/p dataset.
clustered_toronto = pd.merge(df_data, neighborhood_top_clustered_venues, on='Neighborhood', how='inner')


# In[93]:


neighborhood_top_clustered_venues.head()


# In[94]:


# Retrieve Lati and Long of Toronto for Map creation.
address = 'Toronto, Canada'

geolocator = Nominatim(user_agent="explorer")
location = geolocator.geocode(address)
toronto_lati = location.latitude
toronto_longi = location.longitude
print('The geograpical coordinate of {} are {}, {}.'.format(address, toronto_lati, toronto_longi))


# In[95]:


# create clustered map of Toronto using latitude and longitude values
map_toronto_clustered = folium.Map(location=[toronto_lati, toronto_longi], zoom_start=10)

latitude = clustered_toronto['Latitude']
longitude = clustered_toronto['Longitude']
borough = clustered_toronto['Borough']
neighborhood = clustered_toronto['Neighborhood']
clusters = clustered_toronto['Cluster Labels']

colors = ['cyan', 'blue', 'green', 'yellow', 'red']
# add markers to map
for lat, lng, boro, neigh, cluster in zip(latitude, longitude, borough, neighborhood, clusters):
    label = '{}, {}, {}'.format(boro, neigh, cluster)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=4,
        popup=label,
        color=colors[cluster],
        fill=True,
        fill_color=colors[cluster],
        fill_opacity=0.7,
        parse_html=False).add_to(map_toronto_clustered)


# In[97]:


map_toronto_clustered


# In[ ]:


# Create a dataframe with top 10 venues per cluster.


# In[225]:


one_hot = neighborhood_onehot.copy(deep=True)
one_hot.insert(1, 'Cluster Labels', kmeans.labels_)
one_hot.head()


# In[226]:


# Edit dataframe to groupby Clusters
one_hot = one_hot.drop(['Neighborhood'], axis=1)
one_hot.head()


# In[227]:


# Groupby ClusterLables and take mean of all venues available
grouped_one_hot = test_df.groupby('Cluster Labels').mean().reset_index()
grouped_one_hot.head()


# In[228]:


# Create an Empty dataframe which shows top n venue categories in a cluster label.
num_top_venues = 10

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['Cluster Labels']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
cluster_top_venues = pd.DataFrame(columns=columns)
cluster_top_venues['Cluster Labels'] = grouped_test_df['Cluster Labels']

cluster_top_venues.head()


# In[229]:


# Add values to the above dataframe.
def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)    
    return row_categories_sorted.index.values[0:num_top_venues]

for row in np.arange(grouped_test_df.shape[0]):
    cluster_top_venues.iloc[row, 1:] = return_most_common_venues(grouped_test_df.iloc[row, :], num_top_venues)
    
cluster_top_venues


# In[230]:


# Add colors as used in map to Cluster labels.
for i in range(len(cluster_top_venues['Cluster Labels'])):
    cluster_top_venues['Cluster Labels'][i] = str(cluster_top_venues['Cluster Labels'][i]) + '-' + colors[i].upper()

cluster_top_venues.head()


# In[ ]:




