from mpi4py import MPI
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Toplam sayı
n = 10**7  # 10 milyon

# Her işlem parçasına düşen eleman sayısı (n'i işlemci sayısına bölüyoruz)
data_per_process = n // size

# Her işlemci kendi veri aralığını hesaplıyor
start = rank * data_per_process
end = start + data_per_process

# Veriyi bu işlemci için oluştur
data = list(range(start, end))

start_time = time.time()

# Her işlemci kendi verisinin toplamını hesaplıyor
partial_sum = sum(data)

# Tüm işlemcilerin partial_sum değerleri root=0'da toplanır
total_sum = comm.reduce(partial_sum, op=MPI.SUM, root=0)

end_time = time.time()

# Sonuçlar sadece ana işlemcide (rank == 0) yazdırılır
if rank == 0:
    print(f"Processes: {size}")
    print(f"Total computation time: {end_time - start_time:.4f} seconds")
    print(f"Final result: {total_sum}")