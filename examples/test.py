import threading

def add_ops(n):
  while True:
    n = n + 1
    if n == 10:
      break
    
def print_ops(n):
  while True:
    print(n)
    
if __name__ == "__main__":
  task_tds = []
  n = 0
  task_tds.append(add_ops, args=(n))
  task_tds.append(print_ops, args=(n))
  for td in task_tds:
    td.start()
    
  for td in task_tds:
    td.join()
