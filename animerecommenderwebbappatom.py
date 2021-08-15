import streamlit as st
import pandas as pd
import pickle
st.write("""
# Anime reccomendation Web app


This app recommends animes based on the animes you liked
""")
cluster_data=pd.read_csv('top100animesearchcluster1.csv')
animes_df=pd.read_csv('animelist1.csv')
def getdataformodel(user_animes,animes_df):
    useranimelist=user_animes.split(", ")
    animedict=list(animes_df.columns)
    animes_df_c=animes_df.copy()
    animes_df_c.insert(loc=0, column='user_id', value=[1])
    for ele in useranimelist:
        if ele in animedict:
            animes_df_c.loc[0,ele]=1
    pca_transformer = pickle.load(open("pca_animerecommender1.pkl",'rb'))
    output=pca_transformer.transform(animes_df_c)
    return output
def clusterprediction(user_animes,animes_df):
    input=getdataformodel(user_animes,animes_df)
    KNNmodel=pickle.load(open("animeclusterer1.pkl",'rb'))
    pred=KNNmodel.predict(input[:,1:])
    return int(pred)
def creategenrelist(cluster_data):
    genrelist=dict()
    for i in range(400):
        l=cluster_data["genre"][i]
        l=l.split(", ")
        for ele in l:
            if ele not in genrelist:
                genrelist[ele]=""
    genrelist=list(genrelist.keys())
    genrelist.append("All")
    return genrelist
def genrewisereccomendations(user_animes,animes_df,genre,cluster_data):
    cluster=clusterprediction(user_animes,animes_df)
    if genre=="All":
        df=cluster_data[cluster_data["cluster"]==cluster]
        df=df[["name","genre"]]
        return df[0:10]
    else :
        df1=cluster_data[cluster_data["cluster"]==cluster].copy()
        for ind in df1.index:
            l=df1['genre'][ind]
            l=l.split(", ")
            if genre not in l:
                df1.drop(ind,axis=0,inplace=True)
        df1=df1[["name","genre"]]
        return df1[0:10]
st.sidebar.header('Type the animes you like seperated by comma(names as on Myanimelist)')
st.sidebar.write('Sample input -Death Note, Fullmetal Alchemist, Neon Genesis Evangelion')
user_animes= st.sidebar.text_input('Input Animes here')
genrelist=creategenrelist(cluster_data)
genre=st.selectbox('Select the genre of the reccomendations',genrelist)
output_df=genrewisereccomendations(user_animes,animes_df,genre,cluster_data)
output_df=output_df[["name","genre"]]
st.dataframe(data=output_df)
