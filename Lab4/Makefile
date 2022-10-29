  # Compiler for C++
  CC = g++

  # Package Manager for opencv4
  PKG_MNG = `pkg-config --cflags --libs opencv4` -lpthread

  # Compiler flags:
  #  -g    			adds debugging information to the executable file
  #  -Wall 			turns on most, but not all, compiler warnings
  #  -std=c++11 	implements all of the major features of the C++11 standard
  CFLAGS  = -std=c++11 -g -Wall -O0

  # Target executables:
  TARGET = main
  LIB = pthread_barrier

  all: clean lib src compile

  time: clean lib src compile time

  default: clean_target src compile

  pi: clean
	$(CC) -o $(TARGET) $(TARGET).cpp $(PKG_MNG) $(CFLAGS)

  lib: $(LIB).cpp
	$(CC) -c -o $(LIB).o $(LIB).cpp $(CFLAGS)

  src: $(TARGET).cpp
	$(CC) -c -o $(TARGET).o $(TARGET).cpp `pkg-config --cflags opencv4` $(CFLAGS)

  compile: $(TARGET).o $(LIB).o
	$(CC) -o $(TARGET) $(TARGET).o $(LIB).o $(PKG_MNG) $(CFLAGS)

  run: $(TARGET)
	./$(TARGET) "../Media/small_frog.mp4"

  time: $(TARGET)
	time ./$(TARGET) "../Media/small_frog.mp4"

  clean_target:
	$(RM) $(TARGET) $(TARGET).o

  clean:
	$(RM) $(TARGET) $(TARGET).o $(LIB).o
