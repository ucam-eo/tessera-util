from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
name = MPI.Get_processor_name()

print("Hello from rank {} on {}".format(rank, name))

# 检查是否至少有两个进程
if size < 2:
    if rank == 0:
        print("需要至少两个进程来测试节点间通信。")
else:
    if rank == 0:
        # rank 0 向 rank 1 发送消息
        msg = "Hello from rank 0"
        print("Rank 0 sending message to rank 1: {}".format(msg))
        comm.send(msg, dest=1, tag=100)
        # 接收来自 rank 1 的回复
        reply = comm.recv(source=1, tag=200)
        print("Rank 0 received reply from rank 1: {}".format(reply))
    elif rank == 1:
        # rank 1 接收来自 rank 0 的消息
        msg = comm.recv(source=0, tag=100)
        print("Rank 1 received message from rank 0: {}".format(msg))
        # 向 rank 0 发送回复
        reply = "Hello from rank 1"
        comm.send(reply, dest=0, tag=200)
        print("Rank 1 sent reply to rank 0: {}".format(reply))
