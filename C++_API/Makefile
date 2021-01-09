CXXFLAGS += -I $(LIBJIT_DIR)/include/ -Wall -Wextra -Werror
LDFLAGS += $(LIBJIT_DIR)/jit/.libs/libjit.a $(LIBJIT_DIR)/jitplus/.libs/libjitplus.a -lpthread

all: main.cpp
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS)

clean:
	rm -f a.out
