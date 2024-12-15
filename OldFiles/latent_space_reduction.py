from utils import *
import numpy as np

features_label = labels_columns

df_s, df_l = load_data()
embed_s = np.array(df_s[embedding_columns])
embed_l = np.array(df_l[embedding_columns])

def projection(V, dataset):
    return V@dataset.T/np.linalg.norm(V)


new_s = df_s[id_columns+image_name_columns].copy()
new_l = df_l[id_columns+image_name_columns].copy()

new_columns = []
for label in features_label:
    print(label)
    
    indices_label = df_l[label]==1
    indices_not_label = df_l[label]==-1

    l_v_label = np.array(np.sum(df_l[embedding_columns][indices_label]))
    new_l["embed_"+label] = projection(l_v_label, embed_l)


    l_v_not_label = np.sum(df_l[embedding_columns][indices_not_label]) 
    new_l["embed_not_"+label] = projection(l_v_not_label, embed_l)
  

    s_v_label = np.sum(df_s[embedding_columns][indices_label])
    new_s["embed_"+label] = projection(s_v_label, embed_s)

    s_v_not_label = np.sum(df_s[embedding_columns][indices_not_label])
    new_s["embed_not_"+label] = projection(s_v_not_label, embed_s)

    new_columns.append("embed_"+label)
    new_columns.append("embed_not_"+label)


new_l.to_csv("datasets/l_embed.csv")
new_s.to_csv("datasets/s_embed.csv")
