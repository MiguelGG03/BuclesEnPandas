import pandas as pd
import numpy as np

import perfplot 
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})

# Techniques
def forloop(df,col1,col2):
    total = []
    for index in range(len(df)):
        total.append(df[col1].iloc[index]
                    + df[col2].iloc[index])
    return total

def itertuples(df):
    total = []
    for row in df.itertuples():
        total.append(row[1] + row[2])
    return total

def iterrows(df,col1,col2):
    total = []
    for index, row in df.iterrows():
        total.append(row[col1]
                   + row[col2])
    return total

def apply(df,col1,col2):
    return df.apply(lambda row: row[col1]
                              + row[col2], axis=1).to_list()

def comprehension(df,col1,col2):
    return [src + dst for src, dst in zip(df[col1], df[col2])]

def pd_vectorize(df,col1,col2):
    return (df[col1] + df[col2]).to_list()

def np_vectorize(df,col1,col2):
    return (df[col1].to_numpy() + df[col2].to_numpy()).tolist()



def main():
    df = pd.read_csv('https://raw.githubusercontent.com/mlabonne/how-to-data-science/main/data/nslkdd_test.txt')

    # Path: script.py
    print("Digamos que queremos crear una nueva característica:\n"
          "El número total de bytes en la conexión. Solo tenemos que resumir dos características\n"
          "existentes: src_bytes y  dst_bytes. Veamos diferentes métodos para calcular esta nueva\n"
          "característica.")

    print(df.info())
    _cl1= df["dst_host_same_srv_rate"]
    _cl2= df["dst_host_diff_srv_rate"]
    eleccion = input("Que metodo deseas usar?\n(6) Vectorización de pandas (x1500 mas rapido)✅\n(7) Vectorización NumPy (x1900 mas rapido)✅✅\n(8) Perfplot (Bonus)✅✅📈\n>>> ")
    if eleccion == "8":
        # Perfplot
        functions = [iterrows, forloop, apply, itertuples, comprehension, pd_vectorize, np_vectorize]

        df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})

        out = perfplot.bench(
            setup=lambda n: pd.concat([df]*n, ignore_index=True),
            kernels=functions,
            labels=[str(f.__name__) for f in functions],
            n_range=[2**n for n in range(20)],
            xlabel='Number of rows',
        )
    
    elif eleccion=="6":
        # Vectorización de pandas
        print("Vectorización de pandas")
        resultado = pd_vectorize(df,_cl1,_cl2)
        plt.figure(figsize=(20,12))
        plt.plot(resultado)
        plt.show()

        
    elif eleccion=="7":
        # Vectorización NumPy
        print("Vectorización NumPy")
        resultado = np_vectorize(df,_cl1,_cl2)
        plt.figure(figsize=(20,12))
        plt.plot(resultado)
        plt.show()

if __name__=="__main__":
    main()
    