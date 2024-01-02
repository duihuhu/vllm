#include <sys/mman.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/time.h>
int main(void)
{
  // assume file exists
  struct timeval mp_dp_tv;
  int mp_dp_fd = -1;
  if ((mp_dp_fd = open("mprefill_to_mdispatcher.txt", O_RDWR, 0)) == -1)
     {
     printf("unable to open mprefill_to_mdispatcher.txt\n");
     return 0;
     }
  // open the file in shared memory
  char* mprefill_shared = (char*) mmap(NULL, 35 * 1024, PROT_READ | PROT_WRITE, MAP_SHARED, mp_dp_fd, 0);
  // periodically read the file contents
  unsigned char mprefill_num = 0x00;
  unsigned char prefilled_request_num = 0x00;
  int alread_num = 0;

  struct timeval md_dp_tv;
  int md_dp_fd = -1;
  if ((md_dp_fd = open("mprefill_mdispatcher_to_mdecode_mdispatcher.txt", O_RDWR, 0)) == -1)
     {
     printf("unable to open mprefill_mdispatcher_to_mdecode_mdispatcher.txt\n");
     return 0;
     }
  // open the file in shared memory
  char* dispatcher_shared = (char*) mmap(NULL, 35 * 1024, PROT_READ | PROT_WRITE, MAP_SHARED, md_dp_fd, 0);
  // periodically read the file contents

  // execute 
  int flag = 0;
  while (1)
  {
    if (mprefill_num!=mprefill_shared[alread_num*35])
    {
      // mprefill_num = mprefill_shared[0];
      // prefilled_request_num = mprefill_shared[1];
      // gettimeofday(&mp_dp_tv, NULL);
      // long long timestamp_microseconds = (long long)mp_dp_tv.tv_sec * 1000000 + mp_dp_tv.tv_usec;
      // printf("Current timestamp: 0x%02X, 0x%02X, %lld \n", mprefill_num, prefilled_request_num , timestamp_microseconds);

      // lseek(md_dp_fd, 0, SEEK_SET);
      // dispatcher_shared[0] = mprefill_num;
      // dispatcher_shared[1] = prefilled_request_num;

      memcpy(dispatcher_shared+(alread_num*35), mprefill_shared+(alread_num*35), 35);
      alread_num = alread_num + 1;
    }
  } 

   return 0;
}