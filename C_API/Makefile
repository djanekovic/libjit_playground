CXXFLAGS += -I $(LIBJIT_DIR)/include/ -Wall -Wextra -Werror
LDFLAGS += $(LIBJIT_DIR)/jit/.libs/libjit.a -lpthread

all: main.cpp
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS)

clean:
	rm -f a.out
