# plot
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as patches

# data
import pandas as pd
import geopandas as gpd

# world map 
world = gpd.read_file("ne_110m_admin_0_countries.shp")

# filter on europe only
europe = world[(world['CONTINENT'] == 'Europe') | (world['NAME'] == 'Turkey')]


rep = {
    'Country': ['Turkey', 'Albania', 'Russia','Iceland', 'Ukraine', 'Belarus', 'Bosnia and Herz.', 'Bulgaria', 'Czechia', 'Greece', 'Croatia', 'Hungary', 'Ireland', 'Italy', 'Luxembourg', 'Moldova', 'Montenegro', 'North Macedonia', 'Portugal', 'Romania', 'Serbia', 'Kosovo', 'Slovenia', 'Slovakia', 'Austria', 'Germany', 'Sweden', 'Netherlands', 'Spain', 'France', 'Denmark', 'Switzerland', 'Belgium', 'Poland', 'Estonia', 'Lithuania', 'Finland', 'Latvia', 'Norway', 'United Kingdom'],
    'Representation': [4,4,4,4,4,4,4,4,4,4,4,4,4,4,1,4,4,4,4,4,4,4,4,4,2 ,1 ,1 ,2 ,2 ,2 ,2 ,2 ,2 ,1 ,1 ,1 ,3 ,1 ,2 ,2 ] # NEBB - power system and DH - v3
}

# tuple list (country, representation)
rep_list = [
    ("Turkey", 4),
    ("Albania", 4),
    ("Russia", 4),
    ("Iceland", 4),
    ("Ukraine", 4),
    ("Belarus", 4),
    ("Bosnia and Herz.", 4),
    ("Bulgaria", 4),
    ("Czechia", 4),
    ("Greece", 4),
    ("Croatia", 4),
    ("Hungary", 4),
    ("Ireland", 4),
    ("Italy", 4),
    ("Moldova", 4),
    ("Montenegro", 4),
    ("North Macedonia", 4),
    ("Portugal", 4),
    ("Romania", 4),
    ("Serbia", 4),
    ("Kosovo", 4),
    ("Slovenia", 4),
    ("Slovakia", 4),
    ("Luxembourg", 1),
    ("Austria", 2),
    ("Germany", 1),
    ("Sweden", 1),
    ("Netherlands", 2),
    ("Spain", 2),
    ("France", 2),
    ("Denmark", 2),
    ("Switzerland", 2),
    ("Belgium", 2),
    ("Poland", 1),
    ("Estonia", 1),
    ("Lithuania", 1),
    ("Finland", 3),
    ("Latvia", 1),
    ("Norway", 2),
    ("United Kingdom", 4)
]


df_rep = pd.DataFrame(rep_list, columns=["Country", "Representation"])
data = europe.merge(df_rep, how='left', left_on='NAME', right_on='Country')


# initialize the figure
fig, ax = plt.subplots(1, 1, figsize=(8, 8))

# create the plot
#data.plot(ax=ax)
data.plot(column='Representation', ax=ax, cmap='Set2', edgecolor='white', linewidth=0.2) # , legend=True)

# custom axis
ax.set_xlim(-15, 35)
ax.set_ylim(32, 72)
ax.axis('off')

# display the plot
plt.tight_layout()
plt.show()
#plt.savefig("figure.png", dpi=600, bbox_inches='tight')