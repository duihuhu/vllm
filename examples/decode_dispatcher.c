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
  //dispatcher to dispatcher
  struct timeval start;
  struct timeval end;

  struct timeval mp_md_dp_tv;
  int md_md_dp_fd = -1;
  if ((md_md_dp_fd = open("mprefill_mdispatcher_to_mdecode_mdispatcher.txt", O_RDWR, 0)) == -1)
     {
     printf("unable to open mprefill_dispatcher_to_mdecode_dispatcher.txt\n");
     return 0;
     }
  // open the file in shared memory
  char* dispatcher_shared = (char*) mmap(NULL, 35 * 1024, PROT_READ | PROT_WRITE, MAP_SHARED, md_md_dp_fd, 0);
  // periodically read the file contents
  unsigned char mprefill_num = 0x00;
  unsigned char prefilled_request_num = 0x00;

  int alread_num = 0;

  //from mdecode's mdispatcher to decode
  struct timeval dp_md_tv;
  int dp_md_fd = -1;
  if ((dp_md_fd = open("mdispatcher_to_mdecode.txt", O_RDWR, 0)) == -1)
     {
     printf("unable to open mdispatcher_to_mdecode.txt\n");
     return 0;
     }
  // open the file in shared memory
  char* mdecode_shared = (char*) mmap(NULL, 35 * 1024, PROT_READ | PROT_WRITE, MAP_SHARED, dp_md_fd, 0);
  // periodically read the file contents

  // execute 
  int flag = 0;
  while (1)
  {
    if (mprefill_num!=dispatcher_shared[alread_num*35+34])
    {
      // mprefill_num = dispatcher_shared[0];
      // prefilled_request_num = dispatcher_shared[1];
      // gettimeofday(&mp_md_dp_tv, NULL);
      // long long timestamp_microseconds = (long long)mp_md_dp_tv.tv_sec * 1000000 + mp_md_dp_tv.tv_usec;
      // printf("Current timestamp: 0x%02X, 0x%02X, %lld \n", mprefill_num, prefilled_request_num , timestamp_microseconds);
      
      // lseek(dp_md_fd, 0, SEEK_SET);
      // mdecode_shared[0] = mprefill_num;
      // mdecode_shared[1] = prefilled_request_num;
      gettimeofday(&start, NULL);
      long long start_timestamp_microseconds = (long long)start.tv_sec * 1000000 + start.tv_usec;
      memcpy(mdecode_shared+(alread_num*35), dispatcher_shared+(alread_num*35), 35);
      gettimeofday(&end, NULL);
      long long end_timestamp_microseconds = (long long)start.tv_sec * 1000000 + start.tv_usec;
      printf("alread_num %d , %lld\n", alread_num, start_timestamp_microseconds-end_timestamp_microseconds);
      alread_num = alread_num + 1;
    }
  } 

   return 0;
}