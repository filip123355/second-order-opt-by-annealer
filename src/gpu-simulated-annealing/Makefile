NVCC := nvcc
TARGET := engine
SRCS := engine.cu

all: $(TARGET)

$(TARGET): $(SRCS)
	$(NVCC) -O2 -arch=sm_80 -o $(TARGET) $(SRCS) -lcurand

clean:
	rm -f $(TARGET)

.PHONY: all clean
