ifdef DEBUG
CXXFLAGS:=-g3
else
CXXFLAGS:=-O2 -flto -DNDEBUG
endif

OS:=$(shell uname -s)

ifeq ($(OS),Darwin)
# Mac OS
CXX:=g++
DLL=dylib
# Auto-detect Homebrew libomp location (works for both Intel and Apple Silicon)
LIBOMP_PREFIX:=$(shell brew --prefix libomp 2>/dev/null || echo /usr/local)
OPENMPFLAGS:=-I$(LIBOMP_PREFIX)/include -Xpreprocessor -fopenmp
OPENMPLIB:=-L$(LIBOMP_PREFIX)/lib -lomp
else
# Linux
DLL=so
OPENMPFLAGS:=-Xpreprocessor -fopenmp
OPENMPLIB:=-fopenmp
endif

INC:=include  # relative include path (for Eigen library)

DLLOBJS:= utils.o array3d.o basis.o trajectories.o operators.o dqm_python.o

all : bin dqm_python.$(DLL)

dqm_python.$(DLL) : $(DLLOBJS)
	$(CXX) -shared -fpic $(CXXFLAGS) -std=c++14 $(OPENMPLIB) -o $@ $(DLLOBJS)
	mv $@ ../dqm/bin/

%.o : %.cpp
	$(CXX) $(CXXFLAGS) -std=c++14 -c -fpic $(OPENMPFLAGS) -I$(INC) -o $@ $<

bin :
	mkdir -p ../dqm/bin

clean :
	rm -rf *.o ../dqm/bin/*.$(DLL)
