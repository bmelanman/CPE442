  # Compiler for C++
  CC = g++

  # Package Manager for opencv4
  PKG_MNG = `pkg-config --cflags --libs opencv4`

  # Compiler flags:
  #  -g    			adds debugging information to the executable file
  #  -Wall 			turns on most, but not all, compiler warnings
  #  -std=c++11 	implements all of the major features of the C++11 standard
  CFLAGS  = $(PKG_MNG) -std=c++11 -g -Wall

  # Target build executable:
  TARGET = main

  default: $(TARGET).cpp
    # g++ main.cpp -o main `pkg-config --cflags --libs opencv4` -std=c++11 -g -Wall
	$(CC) $(TARGET).cpp -o $(TARGET) $(CFLAGS)  

  pi:
    # g++ main.cpp -o main `pkg-config --cflags --libs opencv4`
	$(CC) $(TARGET).cpp -o $(TARGET) $(PKG_MNG)  

  clean:
	$(RM) $(TARGET)

