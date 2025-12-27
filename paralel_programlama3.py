from mpi4py import MPI
import numpy as np

comm=MPI.COMM_WORLD
rank=comm.Get_rank()
size=comm.Get_size()

m,n,p=5000, 5000, 5000 #Matris boyutları

if rank==0:
    A=np.arange(1,m*n+1).reshape(m,n)#Matris oluştur
    B=np.arange(1,n*m+1).reshape(n,p)
else:
    A=None
    B=None

B=comm.bcast(B,root=0)#Çarpılacak matris tüm threadlere paylaşır

rows_per_process=m//size#Thread sayısına göre işlem bölümü hesaplanır
local_A=np.zeros((rows_per_process,n))#Her thread için bir boş dizi 
local_C=np.zeros((rows_per_process,p))

comm.Scatter(A,local_A,root=0)#İşlem yapılacak olan threadlere veri paylaşımını gerçekleştirir

start_time=MPI.Wtime()

for i in range(rows_per_process):
    for j in range(p):
        local_C[i,j]=np.dot(local_A[i,:],B[:,j])#matris çarpım işlemi

C=None
if rank==0:
    C=np.zeros((m,p))#Sadece ana thread içinde toplam sonuçları tutması için bir 

comm.Gather(local_C,C,root=0)#Tüm threadlere yapılan işlemlerin sonuçlarını

end_time=MPI.Wtime()

if rank==0:
    print("Matrix A:")
    print(A)
    print("Matrix B:")
    print(B)
    print("Matrix C(A*B):")
    print(C)
    print(f"Execution time with {size}process:{end_time-start_time:.6f}seconds")