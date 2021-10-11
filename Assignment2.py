# In[44]:

import matplotlib.pyplot as plt
import mplleaflet
import pandas as pd

def leaflet_plot_stations(binsize, hashid):

    df = pd.read_csv('data/C2A2_data/BinSize_d{}.csv'.format(binsize))

    station_locations_by_hash = df[df['hash'] == hashid]

    lons = station_locations_by_hash['LONGITUDE'].tolist()
    lats = station_locations_by_hash['LATITUDE'].tolist()

    plt.figure(figsize=(8,8))

    plt.scatter(lons, lats, c='r', alpha=0.7, s=200)

    return mplleaflet.display()

leaflet_plot_stations(400,'6b179b701cddfffbf4dc1b9bcd8758b42f034daacafa6cf9c09e8931')


# In[60]:

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
df = pd.read_csv('data/C2A2_data/BinnedCsvs_d400/6b179b701cddfffbf4dc1b9bcd8758b42f034daacafa6cf9c09e8931.csv')
df = df.sort_values(by=['Date', 'ID'])
df.info()


# In[61]:

df.head(10)


# In[62]:

df['Year'] = df['Date'].apply(lambda x: x[:4])
df['Date'] = df['Date'].apply(lambda x: x[-5:])
df = df[df['Date'] != '02-29']
df_before_2015 = df[~(df['Year'] == '2015')]
df_2015 = df[df['Year'] == '2015']
df_before_2015.head()


# In[63]:

max_b_2015 = df_before_2015.groupby('Date').agg({'Data_Value':np.max})
min_b_2015 = df_before_2015.groupby('Date').agg({'Data_Value':np.min})
max_2015 = df_2015.groupby('Date').agg({'Data_Value':np.max})
min_2015 = df_2015.groupby('Date').agg({'Data_Value':np.min})
all_max = pd.merge(max_b_2015.reset_index(), max_2015.reset_index(), left_index=True, on = 'Date')
all_min = pd.merge(min_b_2015.reset_index(), min_2015.reset_index(), left_index=True, on = 'Date')


# In[64]:

record_max = all_max[all_max['Data_Value_y'] > all_max['Data_Value_x']]
record_min = all_min[all_min['Data_Value_y'] < all_min['Data_Value_x']]
record_max.head()


# In[88]:

get_ipython().magic('matplotlib inline')
# import numpy as np 
# import matplotlib.pyplot as plt

plt.figure(figsize=(15, 10))
plt.plot(max_b_2015.values, c='orange', label='High Temp Record')
plt.plot(min_b_2015.values, c='blue', label='Low Temp Record')

plt.scatter(record_max.index.tolist(), record_max['Data_Value_y'].values, c = 'red', label = "Broken high temp in 2015")
plt.scatter(record_min.index.tolist(), record_min['Data_Value_y'].values, c = 'green', label = "Broken low temp in 2015")

plt.xlabel('Day', fontsize=16)
plt.ylabel('Temperature', fontsize=16)
plt.title('Ten Year Temperature (2004-2014) that was broken in 2015', fontsize=25)

plt.legend(loc=8, frameon=False, fontsize=16)
plt.gca().fill_between(range(len(min_b_2015)), 
                       np.array(max_b_2015.values.reshape(len(min_b_2015.values),)), 
                       np.array(min_b_2015.values.reshape(len(min_b_2015.values),)), 
                       facecolor='blue', 
                       alpha=0.20)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.show()


# In[ ]:



