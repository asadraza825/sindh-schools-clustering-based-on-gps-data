import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from   sklearn.cluster import KMeans


#elbow_method
def elbow_method(lat,log):
    K_clusters = range(1,10)
    kmeans = [KMeans(n_clusters=i) for i in K_clusters]
    Y_axis = np.array(lat).reshape(-1,1)
    X_axis = log
    score = [kmeans[i].fit(Y_axis).score(Y_axis) for i in range(len(kmeans))]
    # Visualize
    plt.plot(K_clusters, score)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Score')
    plt.title('Elbow Curve')
    plt.show()
  

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
    gps_list = np.array(gps_list)   
    elbow_method(lat,log)
    plt.scatter(lat,log)
    plt.show()
    
    #Training
    total_clusters =  5
    c = KMeans(n_clusters = total_clusters,init='k-means++')
    #predicted = c.fit_predict(gps_list)
    predicted = c.fit(gps_list)
    
    plt.scatter(lat,log, s =50, c='g')
    for i in predicted.cluster_centers_:
        plt.scatter(i[0],i[1] , s=500, c="r", marker="x")
    
    plt.show()
    
    labels = predicted.labels_
    index = 0
    clusters = []
    school_name = np.array(school_name)
    for i,sch,sch_id in zip(gps_list,school_name,school_id):
        row = [sch,sch_id,i[0],i[1]," "]
        clusters.append(row)
    hm_list = ['Adbul Wahid','Ali Hyder','Saad Ahmed','Raza Ali']
    
    
    df = pd.DataFrame(clusters,columns=['School Name','SEMIS Code','Latitude','Longitude',"Assigned HM"],index=labels)   
    df.sort_index(inplace=True)
    
    for i in range(len(hm_list)):
         #df.at(i,"Assigned HM",hm_list[i])
         df.at[i,"Assigned HM"] = hm_list[i]
    df.to_csv("school_clustering_report.csv")
except:
    print("")