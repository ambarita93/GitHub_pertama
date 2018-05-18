# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 22:41:09 2018

@author: Toshiba
"""
'''
# ini adalah k-means++ modifikasi.
# bentuk modifikasinya: mengikutkan satu titik (titik depot) 
# di setiap cluster yang dibentuk.
'''
import random as rd
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
#import seaborn as sns
from scipy.special import gamma


def f_euclid(data_kota):
    
    hasil_euclid = []
    l = len(data_kota)
    jmlh = 0
    for i in range(l-1):
        
        #print("data[k+1]",data[k+1],k)
        eucl = np.linalg.norm(data_kota[i+1] - data_kota[i])
        jmlh = jmlh+np.sum(eucl)
    hasil_euclid.append(jmlh)
    return hasil_euclid

def fobj(var_kota):
    z=f_euclid(var_kota)
    return z

def jarak_euclid(a,b):
    return np.linalg.norm(b - a)

def jarak_euclid2(kota,centroid,depot):
    j1 = np.linalg.norm(kota - centroid)
    j2 = np.linalg.norm(kota - depot)
    return j1 + j2

def ambil_data(s):
    k = np.copy(s)
    return k

def initialize(X, K,titik_depot):
    #C = X[np.random.randint(0,pjg - 1)]
    #C=(X[4])
    C=titik_depot
    #C_list=list([X[4]])
    C_list=list([C])
    for k in range(1, K):
        D2 = np.array([min([np.linalg.norm(kota-c)**2 for c in C]) for kota in X])
        D2sum=np.sum(D2)
        probs = D2/D2sum
        
        cumprobs = np.cumsum(probs)
        r = np.random.rand()
        for j,p in enumerate(cumprobs):
            if r < p:
                i = j
                break
        C_list.append((X[i]))
        C=np.array(C_list)
        #print("ini C>>",C)
        
    return C

 
def kmeans(data_kota,k,epsilon,titik_depot):
    # data dari excel, diubah menjadi format numpy
    s = data_kota
    # history_centroids sebenarnya gak dipakai
    history_centroids = []
    # buat ngukur banyak kota 
    pjg = len(s)

    # versi k-means ++
    centroid_awal=initialize(s, k, titik_depot)
    #print("centroid mula2: ",centroid_awal)
    
    history_centroids.append(centroid_awal)
    proto_lama = np.zeros(centroid_awal.shape)
    belongs_to = np.zeros((pjg,1))
    norm = jarak_euclid(centroid_awal,proto_lama)

    iterasi = 0

    while norm > epsilon:
        iterasi += 1
        # selama jarak dari centroid yang baru dengan yang lama masih berbeda jauh (dilihat dari norm) maka while true
        norm = jarak_euclid(centroid_awal,proto_lama)
        
        for index_instance,instance in enumerate(s):
            
            dist_vec = np.zeros((k,1))
            for index_centroidawal,centro in enumerate(centroid_awal):
                #print("(index_centroidawal,centro)",index_centroidawal,centro,index_instance,instance)
                dist_vec[index_centroidawal] = jarak_euclid2(instance,centro,titik_depot)
                #dist_vec[index_centroidawal]=jarak_euclid(instance,centro)
            belongs_to[index_instance] = np.argmin(dist_vec)
 
            
        tmp_centroids = np.zeros((k,2))
        
        for index in range(len(centroid_awal)):
            instance_close = [i for i in range(len(belongs_to)) if belongs_to[i] == index]
            
            
            slist = list(s[instance_close])
            slist.append(titik_depot)
            s_temp = np.array(slist)
            ambil = ambil_data(s_temp)
            proto = np.mean(s_temp,axis = 0)
            tmp_centroids[index,:] = proto
        proto_lama = centroid_awal    
        centroid_awal = tmp_centroids
        history_centroids.append(tmp_centroids)
    #print("centroid:  ",centroid_awal)
    return centroid_awal,history_centroids,belongs_to,ambil

def susun_ulang(data_kelompok,titikdepot):
    temp2 = []
    titik = titikdepot
    for data in data_kelompok:
        datalist = data.tolist()
        if titik in datalist:
            #print("titik depot ada di datalist ini")
            while titik in datalist: # memastikan lagi agar titik titik tidak dobel
                datalist.remove(titik) # walaupun sebenarnya degn if sudah cukup. 
            datalist.insert(0,titik)    # nanti diperiksa lagi bro.
            datalist.append(titik)
            temp2.append(np.array(datalist))
        else:
            datalist.insert(0,titik)
            datalist.append(titik)
            temp2.append(np.array(datalist))
            
    return temp2

def plot_seaborn(data_kota,belongs_to,kl,titik_depot,banyakkluster):
    # ubah gabungan data label (belongs_to) dan data kota ke format pandas dengan DataFrame
   
    x = [xy[0] for xy in data_kota]
    y = [xy[1] for xy in data_kota]
    # menambahkan koordinat kota depot 
    x.append(titik_depot[0])
    y.append(titik_depot[1])
    label = [lbl[0] for lbl in belongs_to]
    temp = []
    data_full = []
    pjg=len(data_kota)
    
    for i in range(banyakkluster):
        temp=[]
        
        for j in range(pjg):
            if belongs_to[j] == i:
                temp.append(data_kota[j])
                
        data_full.append(np.array(temp))

    
    newkl = kl + 1
    label.append(newkl)
    
    plt.show()
    return np.array(data_full)     
    

    
def buat_calon(kota,n,titik):
    kandidat=[]
    kota_acak = np.copy(kota)
    for i in range(n):
        kandidat.append(kandidat)
    #print(candit,"candit")
    for i in range(n):

        np.random.shuffle(kota_acak)
        kta = kota_acak.tolist()
        titik        
        
        while titik in kta:
            kta.remove(titik)
        kta.insert(0,titik)
        kta.append(titik)
        kota = np.array(kta)
        kandidat[i] = np.copy(kota)
        #print("ini candit",candit[i],i,type(candit[i]))
    return kandidat
    
       
def main1():
    ##proses kluster dimulai
    banyak_kluster = 3
    print("banyak kluster: ",banyak_kluster)
    epsilon = 0.000005
    data_excel = pd.read_excel('dataTSPberlin52v2.xlsx')
    smain = np.array(data_excel)
    titik_depot = list(smain[0])
    print("titik_depot::",titik_depot)
    
    temp_hasil_ulang=[]
    temp_s=[]
    m=20 # menentukan perulangan k-means
    #print("titik depot yang digunakan:",titik_depot)
    for j in range(m):
        centroid, History, belong, ambildata = kmeans(smain,banyak_kluster,epsilon,titik_depot)
    
        hasilkluster = plot_seaborn(smain,belong,banyak_kluster,titik_depot,banyak_kluster)
    
        hasil_susun_ulang = susun_ulang(hasilkluster,titik_depot)
        temp_hasil_ulang.append(hasil_susun_ulang)
    # hitung SSE
        ttl=0
        total=[]
        
    
        for i in range(banyak_kluster):
            for data in hasil_susun_ulang[i]:
                eucl = np.linalg.norm(centroid[i] - data)
                ttl = ttl+np.sum(eucl)
            total.append(ttl)
        s_total=sum(total)
        temp_s.append(s_total)
        #temp_s=np.array(temp_s)
        #print(s_total)
    v=np.argmin(temp_s)
    #print(v)
    hasil_ulang_fix=temp_hasil_ulang[v]    
    ##akhir dari proses kluster
    #coba = f_euclid(hasil_susun_ulang[0])
    ## tahap cuckoo search
    for k in hasil_ulang_fix:
        plt.scatter(k[:,0],k[:,-1])
        plt.plot(k[:,0],k[:,-1])
    plt.show()
    
    ##
    return hasilkluster,hasil_ulang_fix,centroid,titik_depot,smain

hasilkluster,hasil_susun_ulang,cen, ti_depot,datakota = main1()


# membentuk kandidat solusi/sarang
t_depot = ti_depot
 
kandidat = []
kandidat_besar = []
for data in hasil_susun_ulang:
    
    calon = buat_calon(data,20,t_depot) # 5: 5 calon solusi untuk setiap kluster
    kandidat.append(calon)

'''
selanjutnya masuk ke cs untuk setiap kandidat[i], i=1,2,3,...

semangat Cuckoo !!!!1111!!!!
'''

def Levy_flight():
    # mantegna algorithm by Xin-She Yang
    beta=1.5
    sigma=((gamma(1+beta)*np.sin(np.pi*beta/2))/(gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta);
    
    u=np.random.randn()*sigma
    v=np.random.randn()
    step=u/abs(v)**(1/beta)
    
    return step

def get_a_cuckoo(cuckoo,fb,n,tikdep):
    # n banyak populasi
    
    interval=[(1/n)*i for i in range(n+1)]
    stepsize=abs(Levy_flight())

    pjg3=len(cuckoo)-1
    cek=pjg3>4 
    for i in range(n):
        if interval[i]<=stepsize<interval[i+1] and cek:
            for r in range(i):
                #print("ini i,r dan cuckoo",i,r,cuckoo)
                ne_cuckoo,ne_fb=two_opt(cuckoo,fb)
                cuckoo=ne_cuckoo
                fb=ne_fb
                               
        elif stepsize>=interval[i+1] and pjg3>=11:
            ne_cuckoo,ne_fb=double_bridge(cuckoo,fb)
            fb=ne_fb
        else:
            ne_cuckoo,ne_fb=emptyit(cuckoo,tikdep,fb)
            fb=ne_fb
            
    return ne_cuckoo
    
def two_opt(kota,fb):
    euc_init=400
    euc_best=fb
    tol=euc_init-euc_best
    k=0
    tol=10

    pjg=len(kota)
    #print("ini kota di get a two_opt",kota)
    pjg2=pjg-1
    #print("pjg2",pjg2)
    cek=set([euc_best])
    indek1,indek2=0,0
    best_arr=kota
    while abs(tol)>=0 and k<280: 
        k=k+1
        p=0
        #pilih indeks sebagai perwakilan dari edge yang akan di-remove
         #inisialisasi
        q=np.array([1,0])
        # mencari indeks awal
        while abs(q[0]-q[1])==1 or abs(q[0]-q[1])==0 or abs(q[0]-q[1])==pjg2:
            
            p=p+1
            #print("ini p di loop while",p,q[0],q[1],indek1,indek2)
            #while indek1==indek2 or abs(indek1-indek2)==1:
            indek1=np.random.randint(pjg2) # ambil bilangan bulat acak dari 0-(banyak kota-1)
            indek2=np.random.randint(pjg2)
            q=[indek1,indek2]
                #print("ini q",q)
            
        if q[0]>=q[1]:
            q.sort()   

        a = kota[0:q[0]+1]
        b = kota[q[1]:q[0]:-1]
        c = kota[q[1]+1:]

        al=a.tolist()
        bl=b.tolist()
        cl=c.tolist()

        hasil=np.array(al+bl+cl)
    
        l=len(hasil)
        total=0
        for i in range(l-1):
        
            h=np.linalg.norm(hasil[i+1]-hasil[i])
            total=total+h
            #print("i",i,h) 
        euc_init=euc_best
        if total<euc_best:
            euc_best=total
            best_arr=hasil
        #tes=np.copy(hasil)
        else:
            euc_best=euc_best
        
        tol=euc_best-euc_init
    # menandai perubahan euc_best. supaya jangan kebanyakan perubahan.
        if tol<0:
            cek.add(euc_best)
            #print(euc_best,"2-opt")
            if len(cek)==9:
                print("cek 9 terpenuhi di 2-opt!")
                break
        k=k+1
    #print("ini k di two-opt",k,tol)
    return best_arr,euc_best            
    #print("sesudah:",hasil,euc_init,euc_best)

def double_bridge(kota,fb):
    euc_init=400
    euc_best=fb
    tol=euc_init-euc_best
    k=0
    cek=set([euc_best]) 
    #print("ini kota di get a cuckoo double-bridge",kota)
    pjg=len(kota)-1 
    best_arr=kota
    while abs(tol)>=0 and k<280: 
        #pilih indeks sebagai perwakilan dari edge yang akan di-remove
        
        # kenapa kurang satu? karena ada kota awal yang ditambahkan di posisi akhir
        acak=np.random.randint(pjg) # ambil bilangan bulat acak dari 0-(banyak kota-1)
        indek= [acak%pjg,(acak+2)%pjg,(acak+4)%pjg,(acak+6)%pjg] # kenapa +2, +4 +8 ? kenapa gak lebih dari 8? sembarang. yang penting tidak lebih dari pjg

        q=np.array(indek)
        q.sort()    
# berdasarkan pergerakan dari double-bridge move
        a = kota[0:q[0]+1]
        b = kota[q[2]+1:q[3]+1]
        c = kota[q[1]+1:q[2]+1]
        d = kota[q[0]+1:q[1]+1]
        e = kota[q[3]+1:]

        al=a.tolist()
        bl=b.tolist()
        cl=c.tolist()
        dl=d.tolist()
        el=e.tolist()

        hasil=np.array(al+bl+cl+dl+el) # hasil adalah rute yang baru yang dibentuk dari kota
        
        l=len(hasil)
        total=0
        for i in range(l-1):
        
            h=np.linalg.norm(hasil[i+1]-hasil[i])
            total=total+h
           
        euc_init=euc_best
        if total<euc_best:
            euc_best=total
            best_arr=hasil
       
        else:
            euc_best=euc_best
        
        tol=euc_best-euc_init
    # menandai perubahan euc_best. supaya jangan kebanyakan perubahan.
        if tol<0:
            cek.add(euc_best)
            #print("cek",cek)
            #print(euc_best,"DB")
            if len(cek)==9:
                print("cek 9 terpenuhi di DB!")
                break
        k=k+1
    
    return best_arr,euc_best

def choose_a_nest(m,Kbest):
    # untuk memilih secara acak sarang yang akan dibandingkan.
    temp=m-1
    k = np.floor(np.dot(np.random.rand(),temp))
    if k==Kbest:
        k=np.mod(k+1,temp)+1
    return k

def get_max_nest(f_max):
    bes2 = np.argmax(f_max)
    return bes2

def get_best_nest(f_best):
    bes1= np.argmin(f_best)
    return bes1 

def emptyit(kota,titik_depot,fb):
    k_list=kota.tolist()
    best_arr=kota
    tol=10
    euc_best=fb
    euc_init=100
    k=0
    cek=set([euc_best]) 
    while tol>0 and k<200:
        while titik_depot in k_list:
            k_list.remove(titik_depot)
        rd.shuffle(k_list)
        k_list.append(titik_depot)
        k_list.insert(0,titik_depot)
        
        hasil=np.array(k_list)
    
    
        l=len(hasil)
        total=0
        
        for i in range(l-1):
        
            h=np.linalg.norm(hasil[i+1]-hasil[i])
            total=total+h
           
        euc_init=euc_best
        if total<euc_best:
            euc_best=total
            best_arr=hasil
       
        else:
            euc_best=euc_best
        
        tol=euc_best-euc_init
    # menandai perubahan euc_best. supaya jangan kebanyakan perubahan.
        if tol<0:
            cek.add(euc_best)
            #print("cek",cek)
            if len(cek)==2:
                #print("cek 3 terpenuhi di emptyit()!",euc_best)
                break
        k=k+1
    
    return best_arr,euc_best
    
    
nilaiterbaik=0
telur2terbaik=[]
for indeks,data in enumerate(kandidat):
    

    print("kluster k-",indeks+1)
    
    nest = data
    
    n = 20 # banyak calon solusi
    #fbest = np.ones((n,1)).dot(10**5)
    f_best=[]
    for egg in nest:
        f=fobj(egg)
        f_best.append(f)
    
    f_best=np.array(f_best)
           
    pa = 0.4
    gen=8 #banyaknya generasi cuckoo
    for j in range(gen):
        
        kbest = get_best_nest(f_best)
        bestnest = nest[kbest]
    
        k = choose_a_nest(n,kbest)
        k = np.int(k)
        
        best_v=f_best[kbest].tolist()
        s = get_a_cuckoo(bestnest,best_v[0],n,t_depot)
       
        fnew = fobj(s)
        

        if fnew < f_best[k]:
            f_best[k] = fnew
            nest[k] = s
            #print("terjadi perubahan")
        print(j,best_v[0])
        for i in range(n):
            acak=np.random.rand()
            
            if acak >= pa: # untuk mendapatkan solusi yang baru, gunakan double bridge.
                
                k_pa = get_max_nest(f_best)
                
                #s=emptyit(nest[i,:],d)
                #nest[k_pa] = bestnest
                #nest[k_pa]=double_bridge(bestnest,best_v[0])
                f_value=f_best[k_pa].tolist()
                #print(nest[k_pa],f_value)
                pjg=len(nest[k_pa])-1
                
                if pjg>4 and pjg<11:
                    nest[k_pa],f_best[k_pa]=two_opt(bestnest,best_v[0])
                    
                    #nest[k_pa],f_best[k_pa]=two_opt(nest[k_pa],f_value[0])
                elif pjg>=11:
                    #nest[k_pa],f_best[k_pa]=double_bridge(nest[k_pa],f_value[0])
                    nest[k_pa],f_best[k_pa]=double_bridge(bestnest,best_v[0])
                    
                else:
                    nest[k_pa],f_best[k_pa]=emptyit(nest[k_pa],t_depot,f_value[0])
                    
    
                #nest[i,:]=s
                
                # = fobj(nest[k_pa])
                #fbest[i]=fobj(s)
            nilaiterbaik=min(f_best)
        

    telur2terbaik.append(bestnest)
    print("hasilnya adalah:",nilaiterbaik)

smain=datakota
city_number=[]
for sales,data in enumerate(telur2terbaik):
    print("rute sales ke-",sales+1)
    city_number=[]
    for d in data:
        for ind,titik in enumerate(smain):
            if list(titik) == list(d):
                city_number.append(ind)
    print("ada sebanyak:",len(city_number)-1," kota (termasuk titik depot/asal)")
    for number in city_number:
        print("kota-",number+1)                
    
for i,telur in enumerate(telur2terbaik):
    x0=[]
    y0=[]
    nama=[]
    x0=[d[0] for  d in telur]
    y0=[d[1] for  d in telur]    
    nama='kluster'+' '+str(i+1)
    plt.scatter(x0,y0,label=nama)
    plt.plot(x0,y0)    

plt.title("setelah dikluster dan dikenakan Cuckoo Search")
plt.legend()
plt.show()          
