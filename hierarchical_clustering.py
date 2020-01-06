import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial import distance_matrix
from scipy.spatial import distance

# data preprocessing
def cleansing(gps_data):
    data = gps_data.replace(0,np.nan)
    data = data.dropna()
    x = data['LATITUDE']
    y = data['LONGITUDE'] 
    school_id = data['SCH_ID']
    school_name = data['SCH_NAME']
    lat_list = x.tolist()
    log_list = y.tolist()
    school_id_list = school_id.tolist()
    sch_name_list = school_name.tolist()
    gps_matrix = zip(lat_list,log_list)
    return [gps_matrix,[lat_list,log_list,school_id_list,school_name]]

gps_dataset = pd.read_csv("Badin2.csv")
talukas = {1:'Matli',2:'Golarchi-S.F.Rao',3:'Tando Bago',4:'Badin',5:'Talhar'}
ucs = {1:{1:'Malhan',2:'Tharee'},2:{1:'Dubi'},3:{1:'Khoski-Tc'},4:{1:'Seerani'},5:{1:'Talhar-Tc'}}
print("Please select taluka:")
print("1= Matli\n2=Golarchi-S.F.Rao\n3=Tando Bago\n4=Badin\n5=Talhar")
taluka = int(input("Please enter taluka: "))

if(taluka>=1 and taluka <=5):
    taluka_name = talukas[taluka]
    for key,uc in ucs[taluka].items():
        print(key," = ", uc)
    uc_num = int(input("Please enter "+taluka_name+" UC: "))
else:
    print("Please enter correct number")  
    
#filter with Tehsil and UC
#school_data = gps_dataset.query('TEHSIL=="Golarchi-S.F.Rao" and UC == "Dubi"')
try:
        
    uc_name = ucs[taluka][uc_num]
    school_data = gps_dataset.query('TEHSIL=="'+taluka_name+'" and UC=="'+uc_name+'"')
    X = cleansing(school_data)
    lat = X[1][0]
    log = X[1][1]
    school_id = X[1][2]
    school_name = X[1][3]
    gps_list = []
    for i,j in zip(lat,log):
        gps_list.append([i,j])
    # Distance Matrix
    coord_mat = pd.DataFrame(gps_list,columns=['lat','log'],index = school_name)
    #distance_mat = pd.DataFrame(distance_matrix(coord_mat.values,coord_mat.values),index=coord_mat.index,columns=coord_mat.index)
    z = linkage(gps_list,
                method='single',    
                                    
                metric='euclidean'
        )                           
                                    
    
    # visualize dendrogram
    plt.figure(figsize=(30, 10))
    dendrogram(z,labels=coord_mat.index)
    plt.show()
except:
    print("")
