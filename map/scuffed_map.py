import branca.colormap as cm
import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import datetime

# Create dataframe of preds
date, fips, pred = [], [], []

with open('submission.csv') as f:
    submission = f.read().strip().split('\n')[1:]

for line in submission:
    line = line[:10] + ',' + line[11:]
    line = line.strip().split(',')
    date.append(line[0])
    fips.append(int(line[1]))
    pred.append(float(line[6]) ** 0.1)
    # pred.append(np.log(float(line[6]) + 10))

preds_df = pd.DataFrame({'date': date, 'fips':fips, 'pred':pred})
print(preds_df[:10])

# Read in shape file
county_map = gpd.read_file('shapefile2/cb_2018_us_county_20m.shp')
county_map['fips'] = (county_map['STATEFP'].astype(str)+ county_map['COUNTYFP'].astype(str)).astype(int)
print(county_map.columns)
print(county_map.head())
print(county_map['fips'][:5])

# Join preds and shape file
joined_df = preds_df.merge(county_map, on='fips')
joined_df = joined_df[['date', 'fips', 'pred', 'geometry']]
print(joined_df.columns)
print(joined_df.head())
print(min(joined_df['fips']))

# Convert time to ns
joined_df['date'] = pd.to_datetime(joined_df['date']).astype(int) / 10**9
joined_df['date'] = joined_df['date'].astype(int).astype(str)

# Time to map
mymap_fix_boundary = folium.Map(zoom_start=4.2, dragging=False, zoom_control=False, location=[30,-102], tiles='cartodbpositron')
mymap_fix_boundary.save(outfile='fix_boundary.html')

# Colormap
max_colour = max(joined_df['pred'])
min_colour = min(joined_df['pred'])
cmap = cm.linear.YlOrRd_09.scale(min_colour, max_colour)
joined_df['colour'] = joined_df['pred'].map(cmap)
print('finished colormap')

# Style dictionary
fips_list = joined_df['fips'].unique().tolist()
style_dict = {}
for i in range(len(fips_list)):
    fips = fips_list[i]
    result = joined_df[joined_df['fips'] == fips]
    inner_dict = {}
    for _, r in result.iterrows():
        inner_dict[r['date']] = {'color': r['colour'], 'opacity': 0.7}
    style_dict[str(i)] = inner_dict
print('finished style_dict')

# Features for each county
counties_df = joined_df[['geometry']]
print(counties_df.head())
counties_gdf = gpd.GeoDataFrame(counties_df)
# counties_gdf = counties_gdf.drop_duplicates().reset_index()
print('finished features')

# Create map and add colorbar
from folium.plugins import TimeSliderChoropleth

# slider_map = folium.Map(min_zoom=2, max_bounds=True,tiles='cartodbpositron')
slider_map = mymap_fix_boundary

_ = TimeSliderChoropleth(
    data=counties_gdf.to_json(),
    styledict=style_dict,

).add_to(slider_map)

_ = cmap.add_to(slider_map)
cmap.caption = "Relative predicted numbers of deaths in final submission"
slider_map.save(outfile='TimeSliderChoropleth.html')


