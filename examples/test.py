import threading
n = 0
def add_ops():
  global n
  while True:
    n = n + 1
    if n == 10:
      break
    
def print_ops():
  global n
  while True:
    print(n)
    
if __name__ == "__main__":
  task_tds = []

  task_tds.append(threading.Thread(target=add_ops))
  task_tds.append(threading.Thread(target=print_ops))
  for td in task_tds:
    td.start()
    
  for td in task_tds:
    td.join()
