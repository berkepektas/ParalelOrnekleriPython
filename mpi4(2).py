from mpi4py import MPI
import numpy as np

comm=MPI.COMM_WORLD
rank=comm.Get_rank()
size=comm.Get_size()

#Toplayacağımız dizinin boyunu belirleyelim
N=10**7 #1 milyon eleman

#Sadece root (rank 0)

if rank==0:
    data= np.arange(0,N,dtype=np.int64)# 1 den n e kadar olan sayılar
else:
    data= None #Diğer süreçlerde veri yok

#Her sürecin alacağı verileri tutacak bir NumPy array ayarla
data_per_process=N//size #Her sürece düşecek eleman sayısı
local_data=np.zeros(data_per_process,dtype=np.int64)
#Süreyi ölçmeye başla

#Root süreci,diziyi diğer süreçlere paylaştırır
comm.Scatter(data,local_data,root=0)
start_time=MPI.Wtime()
#Her süreç kendi kısmı üzerinde toplama işlemi yapar 
local_sum=sum(local_data)#Python yerleşik sum fonksiyonu kullanılıyor

#Tüm süreçlerin toplamlarını root sürece gönderir ve birleştirir
total_sum=comm.reduce(local_sum,op=MPI.SUM,root=0)

#Süreyi ölçmeyi bitirir
end_time=MPI.Wtime()

#Sonuçları sadece root (rank 0)sürecinde gösterir
if rank==0:
    print("Total sum:", total_sum)
    print(f"Execution time with {size} processes: {end_time - start_time:.6f} seconds")