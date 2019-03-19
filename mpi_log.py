from mpi4py import MPI 
import array

class MPILogFile(object):
    def __init__(self, comm, filename, mode):
        self.file_handle = MPI.File.Open(comm, filename, mode)
        self.file_handle.Set_atomicity(True)
        self.buffer = bytearray

    def write(self, msg):
        b = bytearray()
        b.extend(map(ord, msg))
        self.file_handle.Write_shared(b)

    def close(self):
        self.file_handle.Sync()
        self.file_handle.Close()

comm = MPI.COMM_WORLD
logfile = MPILogFile(
    comm, "test.log", 
    MPI.MODE_WRONLY | MPI.MODE_CREATE | MPI.MODE_APPEND
)
logfile.write("hello")
