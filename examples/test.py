import threading
n = 0
mutex = threading.Lock()

def add_ops():
  global n
  while True:
    mutex.acquire()
    n = n + 1
    mutex.release()
    if n == 10:
      break
    
def print_ops():
  global n
  while True:
    mutex.acquire()
    print(n)
    mutex.release()
    if n == 10:
      break
    
if __name__ == "__main__":
  task_tds = []

  task_tds.append(threading.Thread(target=add_ops))
  task_tds.append(threading.Thread(target=print_ops))
  for td in task_tds:
    td.start()
    
  for td in task_tds:
    td.join()
