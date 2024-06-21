
# build directory
BUILD_DIR=build
CC=nvcc
CFLAGS=-lnccl

all: directories send_recv

directories: ${BUILD_DIR}

${BUILD_DIR}:
	mkdir -p ${BUILD_DIR}

send_recv:
	$(CC) -o $(BUILD_DIR)/send_recv send_recv.cc $(CFLAGS)
# send:
# 	$(CC) -o $(BUILD_DIR)/send Send_Buffers_threads.cc $(CFLAGS)

# recv:
# 	$(CC) -o $(BUILD_DIR)/recv Recv_Buffers_threads.cc $(CFLAGS)

	
.PHONY: clean

clean:
	rm -rf $(BUILD_DIR)